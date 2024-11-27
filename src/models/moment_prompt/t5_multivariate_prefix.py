"""
Changing T5Attention's forward to support prefix tuning, along with subclassing other classes that
use T5Attention. Changes in T5Attention's forward from are marked with
"# <CHANGE>" and "# </CHANGE>". It's possible that the added logic can be separated as some code
that entirely preceeds the original forward, s.t. we can call super().forward() without code
duplciation. Even better, we might be able to use a pre-hook so that most of this won't be needed.
"""

# source:
# https://github.com/allenai/better-promptability/blob/main/better_promptability/models/t5_with_prefix.py#L260



import torch
from torch import nn
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5Block,
    T5Stack,
    T5ForConditionalGeneration,
)

import matplotlib.pyplot as plt


class T5WithPrefixConfig(T5Config):
    def __init__(
        self, num_prefix=None, reparam=False, reparam_dim=512, no_decoder_self_attn=False,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_prefix = num_prefix
        self.reparam = reparam
        self.reparam_dim = reparam_dim
        self.no_decoder_self_attn = no_decoder_self_attn

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        config_dict, kwargs = T5Config.get_config_dict(*args, **kwargs)
        for field in ("num_prefix", "reparam_dim"):
            assert field not in config_dict
            if field in kwargs:
                config_dict[field] = kwargs.pop(field)
        return config_dict, kwargs


class T5AttentionWithPrefix(T5Attention):
    def __init__(self, config: T5WithPrefixConfig, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.num_prefix = config.num_prefix

        self.config = config

    # fmt: off

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).  # noqa: E501
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"  # noqa: E501
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states


        ################################################
        n_channels = self.n_channels
        _, seq_length, d_model = hidden_states.shape
        batch_size_real = batch_size // n_channels
        num_prefix = self.num_prefix

        if self.config.multivariate_projection == 'attention':
            # hidden_states: bs*channel x seq_length x d_model
            hidden_states_ = hidden_states.reshape(-1, n_channels, seq_length, d_model)
            hidden_states_proj = hidden_states_.transpose(1, 2).reshape(-1, n_channels, d_model)
            attn_output, attn_output_weights = self.shared_prompt_projection['mha'](self.shared_prompt_projection['q'](hidden_states_proj),
                                                                                    self.shared_prompt_projection['k'](hidden_states_proj),
                                                                                    self.shared_prompt_projection['v'](hidden_states_proj))
            attn_output = attn_output.reshape(batch_size_real, seq_length, n_channels, -1).permute(0, 2, 3, 1)
            if self.config.agg == 'mlp':
                shared_prompt_projection_k = self.shared_prompt_projection['agg_key'](attn_output) # bs x channel x d_kv x (num_prefix*n_heads)
                shared_prompt_projection_v = self.shared_prompt_projection['agg_value'](attn_output)

            elif self.config.agg == 'rnn':
                # find the padding 
                assert mask.shape[2] == 1
                arr = torch.squeeze(mask).sum(dim=0)

                arr = torch.where(~arr.bool())[0]
                if len(arr) == 0:
                    idx = 0
                else:
                    idx = torch.min(arr)

                # apply mask
                attn_output = attn_output[:, :, :, idx:]

                # TODO: maybe should use hidden state instead of output, just like what you did with llm
                shared_prompt_projection_k = self.shared_prompt_projection['agg_key'](attn_output.reshape(batch_size, -1, seq_length-idx).permute(0,2,1))[0][:, -1, :].reshape(batch_size_real, n_channels, -1).repeat(1, 1, self.config.num_heads)
                shared_prompt_projection_v = self.shared_prompt_projection['agg_value'](attn_output.reshape(batch_size, -1, seq_length-idx).permute(0,2,1))[0][:, -1, :].reshape(batch_size_real, n_channels, -1).repeat(1, 1, self.config.num_heads)

            elif self.config.agg == 'mha':
                # find the padding 
                assert mask.shape[2] == 1
                arr = torch.squeeze(mask).sum(dim=0)

                arr = torch.where(~arr.bool())[0]
                if len(arr) == 0:
                    idx = 0
                else:
                    idx = torch.min(arr)

                # apply mask
                attn_output = attn_output[:, :, :, idx:]
                shared_prompt_projection_k = self.shared_prompt_projection['agg_key'](attn_output.reshape(batch_size, -1, seq_length-idx).permute(0,2,1).repeat(1, 1, self.config.num_prefix),#*self.num_heads),
                                                                                      attn_output.reshape(batch_size, -1, seq_length-idx).permute(0,2,1),
                                                                                      attn_output.reshape(batch_size, -1, seq_length-idx).permute(0,2,1)
                                                                                      )[0]
                shared_prompt_projection_k = shared_prompt_projection_k.mean(dim=1).reshape(batch_size_real, n_channels, -1).repeat(1, 1, self.config.num_heads)

                shared_prompt_projection_v = self.shared_prompt_projection['agg_value'](attn_output.reshape(batch_size, -1, seq_length-idx).permute(0,2,1).repeat(1, 1, self.config.num_prefix),#*self.num_heads),
                                                                                        attn_output.reshape(batch_size, -1, seq_length-idx).permute(0,2,1),
                                                                                        attn_output.reshape(batch_size, -1, seq_length-idx).permute(0,2,1)
                                                                                        )[0]
                shared_prompt_projection_v = shared_prompt_projection_v.mean(dim=1).reshape(batch_size_real, n_channels, -1).repeat(1, 1, self.config.num_heads)

            shared_prompt_projection_key = shared_prompt_projection_k.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.num_heads).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.num_heads, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)
            shared_prompt_projection_value = shared_prompt_projection_v.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.num_heads).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.num_heads, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)


        elif self.config.multivariate_projection == 'vanilla':
            shared_prompt_projection_key = self.prefix_key
            shared_prompt_projection_value = self.prefix_value

        else:
            raise ValueError('Invalid projection type')
        ################################################


        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # <CHANGE>
        # move this up to not include layer-specific prefix
        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None


        key_states = torch.cat([shared_prompt_projection_key, key_states], dim=2)     # self.prefix_key: bsz, num_heads, num_prefix, d_kv
        value_states = torch.cat([shared_prompt_projection_value, value_states], dim=2)


        if mask is not None:
            prefix_mask = torch.zeros(
                batch_size, 1, mask.size(2), num_prefix*(n_channels), device=hidden_states.device
            )
            mask = torch.cat([prefix_mask, mask], dim=-1)
        # </CHANGE>

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9


        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            # <CHANGE>
            position_bias = torch.cat(
                [
                    torch.zeros(
                        position_bias.shape[:3] + (num_prefix*(n_channels),),
                        device=position_bias.device,
                    ),
                    position_bias,
                ],
                dim=3,
            )
            # </CHANGE>

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)


        # <CHANGE>  moved one line up
        # </CHANGE>
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    # fmt: on


class T5LayerSelfAttentionWithPrefix(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        if not config.no_decoder_self_attn:
            self.SelfAttention = T5AttentionWithPrefix(
                config, has_relative_attention_bias=has_relative_attention_bias
            )


class T5LayerCrossAttentionWithPrefix(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.EncDecAttention = T5AttentionWithPrefix(config, has_relative_attention_bias=False)


class T5BlockWithPrefix(T5Block):
    def __init__(self, config, has_relative_attention_bias=False, layer_number=None):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer[0] = T5LayerSelfAttentionWithPrefix(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer[0].SelfAttention.layer_number = layer_number
        if self.is_decoder:
            self.layer[1] = T5LayerCrossAttentionWithPrefix(config)
            self.layer[1].EncDecAttention.layer_number = layer_number


class T5StackWithPrefixMulti(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        # prefix tuning - reparam 'trick'
        per_layer_dim = self.config.num_heads * self.config.d_kv
        total_dim = self.config.num_layers * 2 * per_layer_dim


        if self.config.prefix_tuning:
            self.prefix_embed = (
                nn.Sequential(
                    nn.Embedding(self.config.num_prefix, per_layer_dim),
                    nn.Linear(per_layer_dim, self.config.reparam_dim),
                    nn.Tanh(),
                    nn.Linear(self.config.reparam_dim, total_dim),
                )
                if self.config.reparam
                else nn.Embedding(self.config.num_prefix, total_dim)
            )
            if self.is_decoder:
                self.prefix_embed_cross = (
                    nn.Sequential(
                        nn.Embedding(self.config.num_prefix, per_layer_dim),
                        nn.Linear(per_layer_dim, self.config.reparam_dim),
                        nn.Tanh(),
                        nn.Linear(self.config.reparam_dim, total_dim),
                    )
                    if self.config.reparam
                    else nn.Embedding(self.config.num_prefix, total_dim)
                )


        self.block = torch.nn.ModuleList(
            [
                T5BlockWithPrefix(self.config, has_relative_attention_bias=bool(i == 0), layer_number=i)
                for i in range(self.config.num_layers)
            ]
        )


        if config.multivariate_projection == 'attention':

            self.shared_prompt_projection_k = nn.Linear(config.d_model, config.d_kv, bias=False)
            self.shared_prompt_projection_q = nn.Linear(config.d_model, config.d_kv, bias=False) 
            self.shared_prompt_projection_v = nn.Linear(config.d_model, config.d_kv, bias=False)

            self.shared_prompt_projection_mha = nn.MultiheadAttention(config.d_kv, num_heads=4, batch_first=True,
                                                                      kdim=config.d_kv, vdim=config.d_kv)

            if config.agg == 'mlp':
                self.shared_prompt_projection_agg_key = nn.Linear(config.num_patches, config.num_prefix*config.num_heads) 
                self.shared_prompt_projection_agg_value = nn.Linear(config.num_patches, config.num_prefix*config.num_heads)
            elif config.agg == 'rnn':
                self.shared_prompt_projection_agg_key = nn.RNN(config.d_kv, config.d_kv*config.num_prefix, batch_first=True)
                self.shared_prompt_projection_agg_value = nn.RNN(config.d_kv, config.d_kv*config.num_prefix, batch_first=True)
            elif config.agg == 'mha':
                self.shared_prompt_projection_agg_key = nn.MultiheadAttention(config.d_kv*config.num_prefix, 1, kdim=config.d_kv, vdim=config.d_kv,
                                                                              batch_first=True)
                self.shared_prompt_projection_agg_value = nn.MultiheadAttention(config.d_kv*config.num_prefix, 1, kdim=config.d_kv, vdim=config.d_kv,
                                                                                batch_first=True)
            else:
                raise ValueError('Invalid aggregation type')


            self.shared_prompt_projection = {
                'k': self.shared_prompt_projection_k,
                'q': self.shared_prompt_projection_q,
                'v': self.shared_prompt_projection_v,
                'mha': self.shared_prompt_projection_mha,
                'agg_key': self.shared_prompt_projection_agg_key,
                'agg_value': self.shared_prompt_projection_agg_value,
            }

        elif config.multivariate_projection == 'vanilla':
            self.input_tokens = torch.arange(self.config.num_prefix*self.config.n_channels)
            reparam_dim = 32
            self.prompt_embed = (
                nn.Sequential(
                    nn.Embedding(self.config.num_prefix*self.config.n_channels, per_layer_dim),
                    nn.Linear(per_layer_dim, reparam_dim),
                    nn.Tanh(),
                    nn.Linear(reparam_dim, total_dim),
                )
            )

        else:
            raise ValueError('Invalid projection type')


    class Transpose(torch.nn.Module):
        def forward(self, x):
            return x.T

    class Sample(torch.nn.Module):
        def forward(self, x):
            return x[:, ::(len(x)//16)]


    def generate_prefix_item(self, input_ids, embedding):
        bsz = input_ids.size(0)
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(input_ids.device)
        prefix = embedding(input_tokens)  # batch, seq, layer * embed * 2
        prefix = prefix.view(
            bsz,
            self.config.num_prefix*self.config.n_channels,
            self.config.num_layers,
            2,
            self.config.num_heads,
            self.config.d_kv,
        )
        prefix = prefix.permute([3, 2, 0, 4, 1, 5])  # 2, num_layers, bsz, num_heads, num_prefix, d_kv
        return prefix[0], prefix[1]

    ################################################

    def forward(self, n_channels=None, input_ids=None, inputs_embeds=None, **kwargs):
        if self.config.multivariate_projection == 'attention':
            for block in self.block:
                for i, layer in enumerate(block.layer):
                    if isinstance(layer, T5LayerSelfAttentionWithPrefix):
                        layer.SelfAttention.shared_prompt_projection = self.shared_prompt_projection
                        layer.SelfAttention.n_channels = n_channels
                    if isinstance(layer, T5LayerCrossAttentionWithPrefix):
                        layer.EncDecAttention.shared_prompt_projection = self.shared_prompt_projection
                        layer.EncDecAttention.n_channels = n_channels

        if self.config.multivariate_projection == 'vanilla':

            prefix_key, prefix_value = self.generate_prefix_item(inputs_embeds, self.prompt_embed)
            kwargs['use_cache'] = False
        ################################################
            prefix_key_cross = prefix_value_cross = [None] * len(prefix_key)
            if self.is_decoder:
                prefix_key_cross, prefix_value_cross = self.generate_prefix_item(
                    input_ids, self.prefix_embed_cross
                )
            for block, k, v, k_cross, v_cross in zip(
                self.block, prefix_key, prefix_value, prefix_key_cross, prefix_value_cross
            ):
                for i, layer in enumerate(block.layer):
                    if isinstance(layer, T5LayerSelfAttentionWithPrefix):
                        layer.SelfAttention.prefix_key = k
                        layer.SelfAttention.prefix_value = v
                        layer.SelfAttention.n_channels = n_channels
                    if isinstance(layer, T5LayerCrossAttentionWithPrefix):
                        layer.EncDecAttention.prefix_key = k_cross
                        layer.EncDecAttention.prefix_value = v_cross
                        layer.EncDecAttention.n_channels = n_channels


        output = super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        self.clean_up()
        return output



    def clean_up(self):
        # For safety, in case other code uses it
        if self.config.prefix_tuning:
            for block in self.block:
                for layer in block.layer:
                    if isinstance(layer, T5LayerSelfAttentionWithPrefix):
                        del layer.SelfAttention.prefix_key
                        del layer.SelfAttention.prefix_value
                    if isinstance(layer, T5LayerCrossAttentionWithPrefix):
                        del layer.EncDecAttention.prefix_key
                        del layer.EncDecAttention.prefix_value


class T5ForConditionalGenerationWithPrefixMulti(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = T5StackWithPrefixMulti(self.encoder.config, self.shared)
        self.decoder = T5StackWithPrefixMulti(self.decoder.config, self.shared)
        self.init_weights()
