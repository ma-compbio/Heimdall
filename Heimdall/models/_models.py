"""Heimdall model."""

import warnings

import torch
import torch.nn as nn
from omegaconf import DictConfig

from Heimdall.cell_representations import CellRepresentation
from Heimdall.datasets import PairedInstanceDataset
from Heimdall.embedding import PositionalEncoding
from Heimdall.utils import get_dtype, instantiate_from_config

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.modules.mha import MHA
except ImportError:
    warnings.warn("`flash-attn` is not available.", UserWarning, stacklevel=1)


class HeimdallModel(nn.Module):
    def __init__(
        self,
        data: CellRepresentation,
        model_config: DictConfig,
        task_config: DictConfig,
    ):
        super().__init__()
        """Heimdall model. Combines language model and task-specific head.

        Args:
            data: Cell representation data object.
            model_config: The language model config.
            task_config: The task config.

        """
        self.encoder = instantiate_from_config(
            model_config,
            data,
        )

        self.num_labels = data.num_tasks
        dim_in = self.encoder.d_encoded

        self.reducer = self.reduction_name = None
        if isinstance(data.datasets["full"], PairedInstanceDataset):
            self.reducer, self.reduction_name = instantiate_from_config(
                task_config.reduction,
                dim_in=dim_in,
                return_name=True,
            )

        self.head = instantiate_from_config(task_config.head_config, dim_in=dim_in, dim_out=self.num_labels)

    def forward(self, inputs, labels=None, attention_mask=None):
        # print(inputs, attention_mask)
        if self.reducer is not None:
            all_cell_inputs = zip(*inputs)
            first_cell_mask, second_cell_mask = attention_mask

            encoded_cells = tuple(
                self.encoder(cell_inputs, attention_mask=cell_mask)
                for cell_inputs, cell_mask in zip(all_cell_inputs, attention_mask)
            )

            encoded = self.reducer(encoded_cells)
        else:
            encoded = self.encoder(inputs, attention_mask)

        outputs = self.head(encoded)

        return outputs


class ExpressionOnly(nn.Module):
    def __init__(
        self,
        data: CellRepresentation,
    ):
        super().__init__()
        """Heimdall model. Combines language model and task-specific head.

        Args:
            data: Cell representation data object.
            model_config: The language model config.
            task_config: The task config.

        """

        self.vocab_size = data.adata.n_vars + 2
        self.float_dtype = data.float_dtype
        _, self.d_encoded = data.adata.shape

    def forward(self, inputs, labels=None, attention_mask=None):
        _, outputs = inputs  # extract expression only

        return outputs.to(get_dtype(self.float_dtype))  # convert to float32?


class CellSentenceModel(nn.Module):
    def __init__(
        self,
        data: CellRepresentation,
        d_model: int,
        pos_enc: str,
        pooling: str,
    ):
        super().__init__()
        """Cell sentence encoder abstraction.

        Must set the following properties:
            self.d_encoded: the dimensionality of the embedding output
            self.cell_sentence_model: nn.Module for encoding sequences

        Args:
            data: Cell representation data object.
            d_model: dimensionality of embedding output
            pos_enc: positional encoding strategy to use
            pooling: type of pooling to use for combining gene tokens into cell embedding

        """
        self.d_encoded = d_model
        self.fc = data.fc

        self.vocab_size = data.adata.n_vars + 2  # <PAD> and <MASK> TODO: data.vocab_size

        # Setting up embedding layers
        if data.fg.d_embedding is not None:
            self.gene_embeddings = instantiate_from_config(data.fg.embedding_parameters)
            if data.fg.frozen:
                print("> Freezing all params in F_g")
                for param in self.gene_embeddings.parameters():
                    param.requires_grad = False
        else:
            self.gene_embeddings = None

        if data.fe.d_embedding is not None:
            self.expression_embeddings = instantiate_from_config(data.fe.embedding_parameters)
        else:
            self.expression_embeddings = None

        # Setting up explicit Positional Encodings
        if self.fc.max_input_length is None or (pos_enc in ("none", "NONE")):
            self.position_embeddings = None
        elif pos_enc == "BERT":
            self.position_embeddings = nn.Embedding(self.fc.max_input_length + 1, d_model)  # +1 cuz of CLS
        elif pos_enc == "sincos":
            self.position_embeddings = PositionalEncoding(d_model, max_len=self.fc.max_input_length + 1)
        elif pos_enc is None:
            self.position_embeddings = None
        else:
            raise ValueError("pos_enc can only be one of: `BERT`, `sincos`, `None`")

        self.metadata_embeddings = instantiate_from_config(data.fc.embedding_parameters)

        # Initialize the [CLS] token as a learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # # Encoder - must be subsequently instantiated by child class
        self.cell_sentence_model = None

    def embed_inputs(self, inputs):
        """Embed inputs using the `Fc` reduce function."""
        identity_inputs, expression_inputs = inputs

        input_embeds = self.fc.reduce(
            identity_inputs,
            self.gene_embeddings,
            expression_inputs,
            self.expression_embeddings,
            self.metadata_embeddings,
        )

        batch_size = identity_inputs.size(0)
        seq_length = input_embeds.size(1)

        # Positional Encoding
        if self.position_embeddings is not None:
            if isinstance(self.position_embeddings, nn.Embedding):
                # BERT-style learned embeddings
                position_ids = torch.arange(
                    seq_length,
                    dtype=torch.long,
                    device=input_embeds.device,
                ).expand((batch_size, -1))
                input_embeds += self.position_embeddings(position_ids)
            else:
                # Sinusoidal encoding
                input_embeds = self.position_embeddings(input_embeds)

        return input_embeds

    def forward(self, inputs, attention_mask=None):
        """LM model.

        Args:
            inputs: This is either integers if IDs or bf16/fp32
                floats for predefined embeddings
            attention_mask: A tensor of shape [batchsize, seqlen] where 1/True
                represents no attention and 0/False represents that attention should be used

        Returns:
            torch.tensor: The predicted outputs before cross entropy loss.

        """
        input_embeds = self.embed_inputs(inputs)

        # Encoder
        outputs = self.cell_sentence_model(input_embeds, attention_mask)
        return outputs


class Average(CellSentenceModel):
    def __init__(
        self,
        data: CellRepresentation,
        d_model: int,
        pos_enc: str,
        pooling: str,
    ):
        if pooling != "mean_pooling":
            raise ValueError("Please ensure that `pooling == 'mean_pooling'`")
        if pos_enc is not None:
            raise ValueError("Please ensure that `pos_enc is None`")

        super().__init__(data, d_model=d_model, pos_enc=pos_enc, pooling=pooling)

        self.cell_sentence_model = AverageEncoder()


class ExpressionWeightedSum(CellSentenceModel):
    def __init__(
        self,
        data: CellRepresentation,
        d_model: int,
        pos_enc: str,
        pooling: str,
    ):
        if pooling != "mean_pooling":
            raise ValueError("Please ensure that `pooling == 'mean_pooling'`")
        if pos_enc is not None:
            raise ValueError("Please ensure that `pos_enc is None`")

        super().__init__(data, d_model=d_model, pos_enc=pos_enc, pooling=pooling)

        self.cell_sentence_model = ExpressionWeightedSumEncoder()

    def forward(self, inputs, attention_mask=None):
        """LM model.

        Args:
            inputs: This is either integers if IDs or bf16/fp32
                floats for predefined embeddings
            attention_mask: A tensor of shape [batchsize, seqlen] where 1/True
                represents no attention and 0/False represents that attention should be used

        Returns:
            torch.tensor: The predicted outputs before cross entropy loss.

        """
        _, expression_inputs = inputs
        input_embeds = self.embed_inputs(inputs)

        # Encoder
        outputs = self.cell_sentence_model(expression_inputs, input_embeds, attention_mask)

        return outputs


class Transformer(CellSentenceModel):
    def __init__(
        self,
        data: CellRepresentation,
        d_model: int,
        pos_enc: str,
        pooling: str,
        nhead: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        use_flash_attn: bool,
        num_encoder_layers: int,
    ):
        """Heimdall transformer model.

        Args:
            data: Cell representation data object.
            config: The transformer config.

        .. code-block:: python

        """
        super().__init__(data, d_model=d_model, pos_enc=pos_enc, pooling=pooling)
        self.use_flash_attn = use_flash_attn

        # # Encoder
        self.cell_sentence_model = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            hidden_dropout_prob=hidden_dropout_prob,
            use_flash_attn=use_flash_attn,
            hidden_act=hidden_act,
            num_encoder_layers=num_encoder_layers,
        )

    def forward(self, inputs, attention_mask=None):
        """LM model.

        Args:
            inputs: This is either integers if IDs or bf16/fp32
                floats for predefined embeddings
            attention_mask: A tensor of shape [batchsize, seqlen] where 1/True
                represents no attention and 0/False represents that attention should be used

        Returns:
            torch.tensor: The predicted outputs before cross entropy loss.

        """
        input_embeds = self.embed_inputs(inputs)

        batch_size, seq_length, _ = input_embeds.size()

        # Concatenate the CLS Token to both the attention mask and the input
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand to match batch size
        input_embeds = torch.cat([cls_tokens, input_embeds], dim=1)
        if attention_mask is not None:
            cls_attention = torch.zeros(
                (batch_size, 1),
                dtype=torch.bool,
                device=attention_mask.device,
            )  # Shape: (batch_size, 1)

            attention_mask = torch.cat([cls_attention, attention_mask], dim=1)  # Shape: (batch_size, seq_len + 1)

        # Encoder
        outputs = self.cell_sentence_model(input_embeds, attention_mask)
        return outputs


class AverageEncoder(nn.Module):
    def forward(self, input_embeds, attention_mask):

        # Encoder
        # take the average of the encoder outputs across the sequence length dimension
        # encoder_output = torch.mean(linear_transform, dim=1)
        valid_mask = ~attention_mask
        expanded_mask = valid_mask.unsqueeze(-1)  # Add an extra dimension for broadcasting

        # Mask the input_embeds
        masked_embeds = input_embeds * expanded_mask

        # Sum the valid (unmasked) embeddings along the sequence dimension
        sum_embeds = masked_embeds.sum(dim=1)
        valid_counts = expanded_mask.sum(dim=1)  # Shape: [batch, 1]
        valid_counts = valid_counts.clamp(min=1)

        # Compute the average, taking into account only the valid values
        masked_avg = sum_embeds / valid_counts

        return masked_avg


class ExpressionWeightedSumEncoder(nn.Module):
    """Implementation of expression-weighted sum encoder used by GenePT-w."""

    def forward(self, expression_inputs, input_embeds, attention_mask):

        valid_mask = ~attention_mask
        expanded_mask = valid_mask.unsqueeze(-1)  # Add an extra dimension for broadcasting
        expanded_expression_inputs = torch.unsqueeze(expression_inputs, dim=2)

        # Mask the input_embeds
        masked_embeds = input_embeds * expanded_mask
        masked_expression_inputs = expanded_expression_inputs * expanded_mask

        # Sum the valid (unmasked) embeddings along the sequence dimension
        weighted_sum = masked_embeds.mul(masked_expression_inputs)

        return weighted_sum


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
        else:
            # convert to bool then invert: PyTorch True(pad) -> flash True(keep)
            src_key_padding_mask = ~src_key_padding_mask.bool()

        batch, seq_len, d_model = x.size()

        x_unpad, indices, cu_seqlens, max_seqlen_in_batch, _ = unpad_input(x, src_key_padding_mask)

        # x: (batch, seq_len, d_model)
        for layer in self.layers:
            x_unpad = layer(x_unpad, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch)

        x = pad_input(x_unpad, indices, batch, seq_len)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        hidden_dropout_prob: float,
        use_flash_attn: bool,
        num_encoder_layers: int,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn

        if self.use_flash_attn:
            self.encoder = FlashTransformerEncoder(
                d_model,
                nhead,
                num_encoder_layers,
                dropout=hidden_dropout_prob,
                activation=hidden_act,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=hidden_dropout_prob,
                activation=hidden_act,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, input_embeds, attention_mask=None):
        return self.encoder(input_embeds, src_key_padding_mask=attention_mask)
