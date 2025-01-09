from functools import partial
from typing import Callable, Optional

import torch
from torch import nn

try:
    from flash_attn.modules.block import Block
    from flash_attn.modules.mha import MHA
    from flash_attn.modules.mlp import Mlp
except ImportError:
    raise ImportError("Please install flash_attn from https://github.com/Dao-AILab/flash-attention")


# Based on https://gist.github.com/kklemon/98e491ff877c497668c715541f1bf478
class FlashAttentionTransformerEncoderLayer(Block):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        norm_first: bool = False,
        batch_first: bool = False,
        activation: Callable = torch.nn.functional.gelu,
        rotary_emb_dim: int = 0,
    ):
        # Note: batch_first doesn't do anything (it's always true)

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        mixer_cls = partial(
            MHA,
            num_heads=nhead,
            use_flash_attn=True,
            rotary_emb_dim=rotary_emb_dim,
        )

        mlp_cls = partial(Mlp, hidden_features=dim_feedforward)

        super().__init__(
            d_model,
            mixer_cls=mixer_cls,
            mlp_cls=mlp_cls,
            resid_dropout1=dropout,
            resid_dropout2=dropout,
            prenorm=norm_first,
        )


class FlashAttentionTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()

        try:
            from flash_attn.bert_padding import pad_input, unpad_input
        except ImportError:
            raise ImportError("Please install flash_attn from https://github.com/Dao-AILab/flash-attention")

        self._pad_input = pad_input
        self._unpad_input = unpad_input

        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, src_key_padding_mask=None):
        batch, seqlen = x.shape[:2]

        if src_key_padding_mask is None:
            for layer in self.layers:
                x = layer(x)
        else:
            x, indices, cu_seqlens, max_seqlen_in_batch, used_seqlens_in_batch = self._unpad_input(
                x,
                ~src_key_padding_mask,
            )

            mixer_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen_in_batch,
            }

            for layer in self.layers:
                intermediate_output = layer(x, mixer_kwargs=mixer_kwargs)
                if layer.prenorm:
                    x, _ = intermediate_output
                else:
                    x = intermediate_output

            x = self._pad_input(x, indices, batch, seqlen)

        return x
