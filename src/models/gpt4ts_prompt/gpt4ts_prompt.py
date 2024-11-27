# sources:
# https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Long-term_Forecasting/models/GPT4TS.py
# https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Long-term_Forecasting/embed.py

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import optim
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from .gpt2_prompt import GPT2Model_prompt

from .common import TASKS
from .layers.embed import DataEmbedding
from .utils.masking import Masking
from .utils.utils import get_anomaly_criterion

from functools import partial
import torch.nn.functional as F


@dataclass
class GPT4TSOutputs:
    backcast: torch.Tensor = None
    forecast: torch.Tensor = None
    timeseries: torch.Tensor = None
    reconstruction: torch.Tensor = None
    mask: torch.Tensor = None
    pretrain_mask: torch.Tensor = None
    anomaly_scores: torch.Tensor = None
    metadata: dict = None
    logits: torch.Tensor = None


class GPT4TS_prompt(nn.Module):
    def __init__(self, configs, reduction='mean'):
        super(GPT4TS_prompt, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.patch_size = configs.patch_len
        self.pred_len = configs.forecast_horizon
        self.enc_in = configs.n_channels
        self.c_out = configs.n_channels
        self.d_ff = configs.d_ff
        self.seq_len = configs.seq_len

        self.transformer_backbone = configs.transformer_backbone
        self.randomly_initialize_backbone = configs.randomly_initialize_backbone
        self.freeze_transformer_backbone = True
        self.stride = configs.patch_stride_len
        self.enable_gradient_checkpointing = configs.enable_gradient_checkpointing

        self.patch_num = (
            self.seq_len + self.pred_len - self.patch_size
        ) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.mask_generator = Masking(mask_ratio=configs.mask_ratio)

        self.d_model = GPT2Config.from_pretrained(self.transformer_backbone).n_embd

        self.enc_embedding = DataEmbedding(
            c_in=1,
            d_model=self.d_model,
            dropout=configs.dropout,
            model_name=self.configs.model_name,
        )

        # Loads a pretrained GPT-2 base model. Mostly based on fpt.py
        if self.randomly_initialize_backbone:
            model_config = GPT2Config.from_pretrained(
                self.transformer_backbone
            )  # Different from fpt.py
            self.gpt2 = GPT2Model_prompt(model_config,
                                         patch_num=self.patch_num-1,
                                         c_in=self.enc_in,
                                         num_prefix=configs.num_prefix,
                                         multivariate_projection=configs.multivariate_projection,
                                         agg=configs.agg,
                                         )
            print(f"Initializing randomly initialized GPT-2.")
        else:
            self.gpt2 = GPT2Model_prompt.from_pretrained(
                self.transformer_backbone,
                output_attentions=True,
                output_hidden_states=True,
                patch_num=self.patch_num-1,
                c_in=self.enc_in,
                num_prefix=configs.num_prefix,
                multivariate_projection=configs.multivariate_projection,
                agg=configs.agg,
            )
            print(f"Initializing pre-trained GPT-2.")

        if self.enable_gradient_checkpointing:
            notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            torch.utils.checkpoint.checkpoint = notfailing_checkpoint
            self.gpt2.gradient_checkpointing_enable()
            print("Enabling gradient checkpointing.")


        self.gpt2.c_in = self.enc_in
        self.c_out = 1



        self.gpt2.h = self.gpt2.h[: configs.gpt_layers]
        print("GPT-2 = {}".format(self.gpt2))

        if self.freeze_transformer_backbone and not self.randomly_initialize_backbone:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if "ln" in name or "wpe" in name or "prompt" in name or "out" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False


        if (
            self.task_name == TASKS.LONG_HORIZON_FORECASTING
            or self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        ):
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.out_layer = nn.Linear(self.d_ff, self.c_out)
        elif (
            self.task_name == TASKS.IMPUTATION
            or self.task_name == TASKS.ANOMALY_DETECTION
        ):
            self.out_layer = nn.Linear(self.d_ff, self.c_out, bias=True)
        elif self.task_name == TASKS.PRETRAINING:
            self.mask_generator = Masking(mask_ratio=configs.mask_ratio)
            self.out_layer = nn.Linear(self.d_ff, self.c_out, bias=True)
        elif self.task_name == TASKS.CLASSIFICATION:
            # raise NotImplementedError

            # original gpt4ts repo
            self.act = F.gelu

            if reduction == "mean":
                self.out_layer = nn.Linear(self.d_model * self.seq_len, configs.num_class)
                self.ln_proj = nn.LayerNorm(self.d_model * self.seq_len)
            elif reduction == "concat":
                self.out_layer = nn.Linear(self.d_model * self.seq_len * self.enc_in, configs.num_class)
                self.ln_proj = nn.LayerNorm(self.d_model * self.seq_len * self.enc_in)

        else:
            raise ValueError(f"Unknown task name: {self.task_name}")



    def forecast(self, x_enc, **kwargs):
        """
        Input:
            x_enc : [batch_size x n_channels x seq_len]
        Returns:
            forecast : [batch_size x n_channels x pred_len]

        Note: GPT4TS has 2 forecasting implementations.
        (1) Long-term-forecasting: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Long-term_Forecasting/models/GPT4TS.py
        (2) In remaining directories: For example in https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Anomaly_Detection/models/GPT4TS.py

        This implementation is based on (2).
        """

        B, L, M = x_enc.shape
        x_enc = x_enc.reshape(
            (-1, 1, x_enc.shape[-1])
        )

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [batch_size x seq_len x n_channels]
        enc_out = enc_out.permute(0, 2, 1)  # [batch_size x n_channels x seq_len]
        enc_out = self.predict_linear(enc_out)  # Along temporal dimension
        enc_out = enc_out.permute(
            0, 2, 1
        )  # [batch_size x (seq_len+pred_len) x n_channels]
        enc_out = torch.nn.functional.pad(enc_out, (0, 768 - enc_out.shape[-1]))
        # [batch_size x (seq_len+pred_len) x d_model]

        dec_out = self.gpt2(
            inputs_embeds=enc_out
        ).last_hidden_state  # [batch_size x (seq_len+pred_len) x d_model]

        dec_out = dec_out[:, :, : self.d_ff]  # [batch_size x (seq_len+pred_len) x d_ff]

        dec_out = self.out_layer(
            dec_out
        )  # [batch_size x (seq_len+pred_len) x n_channels]
        dec_out = dec_out.permute(
            0, 2, 1
        )  # [batch_size x n_channels x (seq_len+pred_len)]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev.repeat(1, 1, self.seq_len + self.pred_len))
        dec_out = dec_out + (means.repeat(1, 1, self.seq_len + self.pred_len))
        # [batch_size x n_channels x (seq_len+pred_len)]

        forecast = dec_out[:, :, -self.pred_len :]
        backcast = dec_out[:, :, : -self.pred_len]

        return GPT4TSOutputs(backcast=backcast.reshape(B, L, backcast.shape[-1]), forecast=forecast.reshape(B, L, forecast.shape[-1]), timeseries=x_enc)

    def reconstruct(self, x_enc, mask=None, **kwargs):
        """
        Input:
            x_enc : [batch_size x n_channels x seq_len]
            mask : [batch_size x seq_len]
        Returns:
            forecast : [batch_size x n_channels x pred_len]
        """
        if mask is None:
            mask = torch.ones_like(x_enc)[:, 0, :]

        mask = mask.unsqueeze(1).repeat(
            1, x_enc.shape[1], 1
        )  # [batch_size x n_channels x seq_len]

        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=2) / torch.sum(mask == 1, dim=2)
        means = means.unsqueeze(2).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=2) / torch.sum(mask == 1, dim=2) + 1e-5
        )
        stdev = stdev.unsqueeze(2).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [batch_size x seq_len x n_channels]
        enc_out = torch.nn.functional.pad(enc_out, (0, 768 - enc_out.shape[-1]))
        # [batch_size x seq_len x d_model]

        dec_out = self.gpt2(
            inputs_embeds=enc_out
        ).last_hidden_state  # [batch_size x seq_len x d_model]

        dec_out = dec_out[:, :, : self.d_ff]  # [batch_size x seq_len x d_ff]

        dec_out = self.out_layer(dec_out)  # [batch_size x seq_len x n_channels]
        dec_out = dec_out.permute(
            0, 2, 1
        )  # [batch_size x n_channels x (seq_len+pred_len)]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev.repeat(1, 1, self.seq_len))
        dec_out = dec_out + (means.repeat(1, 1, self.seq_len))
        # [batch_size x n_channels x (seq_len+pred_len)]

        return GPT4TSOutputs(
            reconstruction=dec_out, timeseries=x_enc, mask=mask[:, 0, :]
        )

    def pretraining(
        self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, **kwargs
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
        mask = mask.to(x_enc.device)

        mask = mask.unsqueeze(1).repeat(
            1, x_enc.shape[1], 1
        )  # [batch_size x n_channels x seq_len]

        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=2) / torch.sum(mask == 1, dim=2)
        means = means.unsqueeze(2).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=2) / torch.sum(mask == 1, dim=2) + 1e-5
        )
        stdev = stdev.unsqueeze(2).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [batch_size x seq_len x n_channels]
        enc_out = torch.nn.functional.pad(enc_out, (0, 768 - enc_out.shape[-1]))
        # [batch_size x seq_len x d_model]

        dec_out = self.gpt2(
            inputs_embeds=enc_out
        ).last_hidden_state  # [batch_size x seq_len x d_model]

        dec_out = dec_out[:, :, : self.d_ff]  # [batch_size x seq_len x d_ff]

        dec_out = self.out_layer(dec_out)  # [batch_size x seq_len x n_channels]
        dec_out = dec_out.permute(
            0, 2, 1
        )  # [batch_size x n_channels x (seq_len+pred_len)]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev.repeat(1, 1, self.seq_len))
        dec_out = dec_out + (means.repeat(1, 1, self.seq_len))
        # [batch_size x n_channels x (seq_len+pred_len)]

        return GPT4TSOutputs(
            reconstruction=dec_out, timeseries=x_enc, pretrain_mask=mask[:, 0, :]
        )

    def detect_anomalies(
        self, x_enc: torch.Tensor, anomaly_criterion: str = "mse", **kwargs
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        anomaly_criterion : str
        """
        outputs = self.reconstruct(x_enc=x_enc)
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

        return GPT4TSOutputs(
            reconstruction=outputs.reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )

    # def classification(self):
    #     raise NotImplementedError

    def classification(self, x_enc, reduction='mean',
                       input_mask=None,
                       **kwargs):

        # channel independent
        x_enc_ = x_enc.reshape(
            (-1, 1, x_enc.shape[-1])
        )

        # embedding
        enc_out = self.enc_embedding(x_enc_, None)  # [B,T,C]
        enc_out = torch.nn.functional.pad(enc_out, (0, 768 - enc_out.shape[-1]))

        # original gpt4ts repo
        output = self.gpt2(inputs_embeds=enc_out, mask_for_agg=input_mask).last_hidden_state

        B, L, M = x_enc.shape

        output = output.reshape(B, L, M, -1)


        # Mean across channels
        if reduction == "mean":
            # [batch_size x n_patches x d_model]
            output = output.mean(dim=1, keepdim=False)
        # Concatenate across channels
        elif reduction == "concat":
            # [batch_size x n_patches x d_model * n_channels]
            output = output.permute(0, 2, 3, 1).reshape(B, L, -1)

        output = self.act(output).reshape(B, -1)
        output = self.ln_proj(output)
        output = self.out_layer(output)

        return GPT4TSOutputs(logits=output)


    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        if (
            self.task_name == TASKS.LONG_HORIZON_FORECASTING
            or self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        ):
            return self.forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.ANOMALY_DETECTION:
            return self.detect_anomalies(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.IMPUTATION:
            return self.reconstruct(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.PRETRAINING:
            return self.pretraining(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classification(x_enc=x_enc, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
