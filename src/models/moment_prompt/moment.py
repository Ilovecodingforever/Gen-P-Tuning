import logging
import warnings
from argparse import Namespace
from copy import deepcopy
from math import ceil

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import T5Config, T5EncoderModel, T5Model
from transformers.models.t5.modeling_t5 import T5Stack


from .common import TASKS
from .base import TimeseriesOutputs
from .embed import PatchEmbedding, Patching
from .revin import RevIN
from .utils.masking import Masking
from .utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)

from typing import Optional


SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


##############################
from .t5_multivariate_prefix import T5StackWithPrefixMulti, T5ForConditionalGenerationWithPrefixMulti



class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(self.dropout(x))
        x = x.flatten(start_dim=2, end_dim=3)
        return x


class ForecastingHead(nn.Module):
    def __init__(
        self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: Optional[torch.Tensor] = None):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x



class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
        reduction: str = "concat",
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        if reduction == "mean":
            self.linear = nn.Linear(d_model, n_classes)
        elif reduction == "concat":
            self.linear = nn.Linear(n_channels * d_model, n_classes)
        else:
            raise ValueError(f"Reduction method {reduction} not implemented. Only 'mean' and 'concat' are supported.")

    def forward(self, x, input_mask: Optional[torch.Tensor] = None):
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        y = self.linear(x)
        return y


class MOMENT(nn.Module):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len

        self.normalizer = RevIN(
            num_features=1, affine=config.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=config.patch_len, stride=config.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            dropout=config.getattr("dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
        self.encoder = self._get_transformer_backbone(config)

        # Frozen parameters
        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)

        if self.freeze_embedder:
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)


    def _update_inputs(
        self, config: Namespace | dict, **kwargs: dict
    ) -> NamespaceWithDefaults:
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            config.d_model is None
            and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            config.d_model = get_huggingface_model_dimensions(
                config.transformer_backbone
            )
            logging.info(f"Setting d_model to {config.d_model}")
        elif config.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone "
                "unless transformer backbone is a Huggingface model."
            )

        if config.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of "
                "['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return config

    def _get_head(self, task_name: str, forecast_horizon) -> nn.Module:
        if task_name == TASKS.RECONSTRUCTION:
            return PretrainHead(
                self.config.d_model,
                self.config.patch_len,
                self.config.getattr("dropout", 0.1),
                self.config.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.FORECASTING:
            num_patches = (
                max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
            ) // self.config.patch_stride_len + 1
            self.head_nf = self.config.d_model * (num_patches + (self.config.num_prefix if self.config.getattr("MPT", False) else 0))
            return ForecastingHead(
                self.head_nf,
                forecast_horizon,
                self.config.getattr("head_dropout", 0.1),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        elif task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("dropout", 0.1),
                reduction = self.config.getattr("reduction", "concat"),
            )
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        if config.getattr("randomly_initialize_backbone", False):
            model_config = T5Config.from_pretrained(config.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(
                f"Initializing randomly initialized transformer from {config.transformer_backbone}."
            )
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                config.transformer_backbone
            )
            logging.info(
                f"Initializing pre-trained transformer from {config.transformer_backbone}."
            )

        transformer_backbone = transformer_backbone.get_encoder()

        model_config = transformer_backbone.config


        if config.getattr("enable_gradient_checkpointing", True):
            from functools import partial
            notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            torch.utils.checkpoint.checkpoint = notfailing_checkpoint
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")


        #######################################################################
        # prefix tuning
        if config.getattr("prefix_tuning", False) or config.getattr("prefix_tuning_multi", False):
            logging.info("Using prefix tuning.")

            model_config = T5Config.from_pretrained(config.transformer_backbone)
            setattr(model_config, 'num_prefix', self.config.getattr("num_prefix", 2))
            setattr(model_config, 'reparam', True)
            setattr(model_config, 'reparam_dim', 32)
            setattr(model_config, 'no_decoder_self_attn', False)
            setattr(model_config, 'seq_len', self.config.seq_len)
            setattr(model_config, 'multivariate_projection', self.config.multivariate_projection)
            setattr(model_config, 'agg', self.config.agg)
            setattr(model_config, 'visualize_attention', self.config.getattr("visualize_attention", False))
            setattr(model_config, 'n_channels', self.config.getattr("n_channels", False))

            num_patches = (max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
                        ) // self.config.patch_stride_len + 1
            setattr(model_config, 'num_patches', num_patches)

            if config.getattr("prefix_tuning_multi", False):
                setattr(model_config, 'prefix_tuning', config.getattr("prefix_tuning", False))
                transformer_backbone = T5ForConditionalGenerationWithPrefixMulti(model_config)

            transformer_backbone = transformer_backbone.from_pretrained(config.transformer_backbone, config=model_config)

            transformer_backbone.enable_input_require_grads()
            transformer_backbone = transformer_backbone.encoder
        #######################################################################

        return transformer_backbone

    def __call__(self, *args, **kwargs) -> TimeseriesOutputs:
        return self.forward(*args, **kwargs)


    def reconstruction(
        self,
        x_enc: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, _ = x_enc.shape

        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # Prevent too short time-series from causing NaNs
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        enc_in = self.patch_embedding(x_enc, mask=mask) 

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)

        if self.config.transformer_type == "encoder_decoder":
            raise NotImplementedError("Encoder-decoder not implemented for prefix T5.")
        else:
            if isinstance(self.encoder, T5StackWithPrefixMulti):
                outputs = self.encoder(n_channels=n_channels, inputs_embeds=enc_in, attention_mask=attention_mask)
            else:
                outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)


        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        #######################################################################
        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
        #######################################################################

        # TODO: should I denormalize the prompt too?
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        if self.config.getattr("debug", False):
            illegal_output = self._check_model_weights_for_illegal_values()
        else:
            illegal_output = None

        #######################################################################
        # get embedding
        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )
        enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
        # [batch_size x n_patches x d_model]
        input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
            1, 1, self.config.d_model
        )
        enc_out = (input_mask_patch_view * enc_out).sum(
            dim=1
        ) / input_mask_patch_view.sum(dim=1)
        #######################################################################


        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=mask,
            illegal_output=illegal_output,
        #######################################################################
            embeddings=enc_out
        #######################################################################
        )

    def forecast(
        self, x_enc: torch.Tensor, input_mask: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
        **kwargs
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)

        if isinstance(self.encoder, T5StackWithPrefixMulti):
            outputs = self.encoder(n_channels=n_channels, inputs_embeds=enc_in, attention_mask=attention_mask)
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)

        # outputs = self.encoder(n_channels=n_channels, inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        #######################################################################
        dec_out = self.head(enc_out)  # [batch_size x n_channels x forecast_horizon]
        #######################################################################
        dec_out = self.normalizer(x=dec_out, mode="denorm")


        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)



    def classify(
        self,
        x_enc: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        reduction: Optional[str] = "concat",
        task_name: Optional[str] = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, _ = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, self.config.seq_len)).to(x_enc.device)
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        if self.categorical_embedding is not None:
            x_enc_new = []
            for i in range(n_channels):
                if str(i) in self.categorical_embedding:
                    assert (x_enc[:, i, :]%1 == 0).all() # check if all values are integers
                    x_enc_new.append(self.categorical_embedding[str(i)](x_enc[:, i, :].long()).permute(0, 2, 1))
                else:
                    x_enc_new.append(x_enc[:, i, :].unsqueeze(1))

            x_enc = torch.cat(x_enc_new, dim=1)
            batch_size, n_channels, _ = x_enc.shape


        x_enc = self.tokenizer(x=x_enc)

        enc_in = self.patch_embedding(x_enc, mask=input_mask) 

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        if patch_view_mask.dim() == 2:
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        elif patch_view_mask.dim() == 3:
            attention_mask = torch.flatten(patch_view_mask, end_dim=-2)

        if self.config.transformer_type == "encoder_decoder":
            raise NotImplementedError("Encoder-decoder not implemented for prefix T5.")
        else:
            if isinstance(self.encoder, T5StackWithPrefixMulti):
                outputs = self.encoder(n_channels=n_channels, inputs_embeds=enc_in, attention_mask=attention_mask)
            else:
                outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)


        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))


        # Mean across channels
        if reduction == "mean":
            # [batch_size x n_patches x d_model]
            enc_out = enc_out.mean(dim=1, keepdim=False)
        # Concatenate across channels
        elif reduction == "concat":
            # [batch_size x n_patches x d_model * n_channels]
            enc_out = enc_out.permute(0, 2, 3, 1).reshape(
                batch_size, n_patches, self.config.d_model * n_channels)

        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        logits = self.head(enc_out, input_mask=input_mask)

        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata=reduction)




    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        task_name: str = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if self.task_name == TASKS.RECONSTRUCTION:
            return self.reconstruction(
                x_enc=x_enc, mask=mask, input_mask=input_mask,
                task_name=task_name,
                **kwargs
            )
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.FORECASTING:
            return self.forecast(x_enc=x_enc, input_mask=input_mask,
                                 task_name=task_name,
                                 **kwargs)
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classify(x_enc=x_enc, input_mask=input_mask,
                                 task_name=task_name,
                                 **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


class MOMENTPipeline(MOMENT, PyTorchModelHubMixin):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        self.new_task_name = kwargs.get("model_kwargs", {}).pop(
            "task_name", TASKS.RECONSTRUCTION
        )
        super().__init__(config, **kwargs)




    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("model_kwargs", {"task_name": TASKS.RECONSTRUCTION})
        kwargs["model_kwargs"].setdefault("task_name", TASKS.RECONSTRUCTION)
        config = Namespace(**kwargs["model_kwargs"])

        if config.task_name == TASKS.FORECASTING:
            if not hasattr(config, "forecast_horizon"):
                raise ValueError(
                    "forecast_horizon must be specified for long-horizon forecasting."
                )

    def init(self) -> None:
        if self.new_task_name != TASKS.RECONSTRUCTION:
            self.task_name = self.new_task_name

        ###################################
        if self.task_name == TASKS.FORECASTING:
            self.head = self._get_head(TASKS.FORECASTING, forecast_horizon=self.config.forecast_horizon)
        elif self.task_name == TASKS.CLASSIFICATION:
            self.head = self._get_head(TASKS.CLASSIFICATION, None)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
        ###################################


def freeze_parameters(model):
    """
    Freeze parameters of the model
    """
    # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model
