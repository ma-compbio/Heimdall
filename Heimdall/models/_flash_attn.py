from torch import nn

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.modules.mha import MHA
except ImportError as e:
    raise ImportError(
        "`flash-attn` is not available. Please install `flash-attn`,"
        " or default to the standard `model=transformer` config.",
    ) from e


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1, activation="gelu"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.activation_fn = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_flash_attn=True,
            causal=False,  # set True if using causal attention
        )

        # LayerNorms (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network
        dim_feedforward = dim_feedforward or d_model * 4
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, cu_seqlens, max_seqlen):
        # x: (batch, seq_len, d_model)
        # attention_mask: (batch, seq_len) with True=keep, False=pad
        # Pre-norm
        x_norm = self.norm1(x)

        # FlashAttention expects mask as (batch, seq_len) where True means keep
        attn_output = self.self_attn(x_norm, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        x = x + self.dropout_layer(attn_output)

        # Feedforward
        x_norm2 = self.norm2(x)
        ff_out = self.linear2(self.activation_fn(self.linear1(x_norm2)))
        x = x + self.dropout_layer(ff_out)

        return x


class FlashTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=None, dropout=0.1, activation="gelu"):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FlashTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
                for _ in range(num_layers)
            ],
        )

    def forward(self, x, src_key_padding_mask=None):
        if src_key_padding_mask is None:
            src_key_padding_mask = None
            x_unpad, cu_seqlens, max_seqlen_in_batch = x, None, None
        else:
            # convert to bool then invert: PyTorch True(pad) -> flash True(keep)
            src_key_padding_mask = ~src_key_padding_mask.bool()

            batch, seq_len, d_model = x.size()

            x_unpad, indices, cu_seqlens, max_seqlen_in_batch, _ = unpad_input(x, src_key_padding_mask)

        # x: (batch, seq_len, d_model)
        for layer in self.layers:
            x_unpad = layer(x_unpad, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch)

        if src_key_padding_mask is None:
            x = x_unpad
        else:
            x = pad_input(x_unpad, indices, batch, seq_len)

        return x
