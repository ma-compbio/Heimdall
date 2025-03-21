"""Heimdall model."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from Heimdall.cell_representations import CellRepresentation
from Heimdall.datasets import PairedInstanceDataset
from Heimdall.utils import get_dtype, instantiate_from_config


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    # predictions: torch.Tensor
    sequence_embeddings: torch.Tensor
    # pooled_embeddings: torch.Tensor
    cls_embeddings: torch.Tensor

    @property
    def device(self):
        return self.logits.device

    def to(self, device):
        for key, val in self.__dict__.items():
            self.__dict__[key] = val.to(device)

    @classmethod
    def reduce(cls, outputs: list["TransformerOutput"], reduction: Callable = torch.sum):
        keys = cls.__dict__["__annotations__"].keys()
        reduced_output = TransformerOutput(
            **{
                key: reduction(
                    torch.stack([getattr(output, key) for output in outputs], axis=0),
                    axis=0,
                )
                for key in keys
            },
        )
        return reduced_output


class HeimdallModel(nn.Module):
    def __init__(
        self,
        data: CellRepresentation,
        model_config: DictConfig,
        task_config: DictConfig,
        conditional_input_types: Optional[dict] = None,
    ):
        super().__init__()
        """Heimdall model. Combines language model and task-specific head.

        Args:
            data: Cell representation data object.
            model_config: The language model config.
            task_config: The task config.
            conditional_input_types: Conditional input types specification.

        """
        self.encoder = instantiate_from_config(
            model_config,
            data,
            conditional_input_types,
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

    def forward(self, inputs, labels=None, conditional_tokens=None, attention_mask=None):
        # handling when there are no conditional tokens supplied
        if conditional_tokens is not None and len(conditional_tokens) == 0:
            conditional_tokens = None

        # print(inputs, attention_mask)
        if self.reducer is not None:
            all_cell_inputs = zip(*inputs)
            first_cell_mask, second_cell_mask = attention_mask

            encoded_cells = tuple(
                self.encoder(cell_inputs, conditional_tokens, attention_mask=cell_mask)
                for cell_inputs, cell_mask in zip(all_cell_inputs, attention_mask)
            )

            encoded = self.reducer(encoded_cells)
        else:
            encoded = self.encoder(inputs, conditional_tokens, attention_mask)

        outputs = self.head(encoded)

        return outputs


class ExpressionOnly(nn.Module):
    def __init__(
        self,
        data: CellRepresentation,
        conditional_input_types: Optional[dict],
    ):
        super().__init__()
        """Heimdall model. Combines language model and task-specific head.

        Args:
            data: Cell representation data object.
            model_config: The language model config.
            task_config: The task config.
            conditional_input_types: Conditional input types specification.

        """

        self.conditional_input_types = conditional_input_types
        self.vocab_size = data.adata.n_vars + 2
        self.float_dtype = data.float_dtype
        _, self.d_encoded = data.adata.shape

    def forward(self, inputs, labels=None, conditional_tokens=None, attention_mask=None):
        _, outputs = inputs  # extract expression only
        return outputs.to(get_dtype(self.float_dtype))  # convert to float32?


class HeimdallTransformer(nn.Module):
    def __init__(
        self,
        data: CellRepresentation,
        conditional_input_types: Optional[dict],
        d_model: int,
        pos_enc: str,
        pooling: str,
        encoder_layer_parameters: dict,
        encoder_parameters: dict,
    ):
        super().__init__()
        """Heimdall transformer model.

        Args:
            data: Cell representation data object.
            config: The transformer config.
            conditional_input_types: Conditional input types specification.

        Example ``conditional_input_types``:

        .. code-block:: python

            conditional_input_types = {
                "binned_gene_expression_embeddings" : {
                    "type": "learned",
                    "vocab_size": 512,
                    }

                "ESM_embeddings" : {
                    "type": "predefined",
                    "vocab_size": -1
                    }
            }

        """
        self.d_encoded = d_model
        self.conditional_input_types = conditional_input_types

        self.fc = data.fc
        self.use_flash_attn = "flash_attention_models" in encoder_parameters.type

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
            raise NotImplementedError("Sine-Cosine Positional Encodings are not implemented yet")
        elif pos_enc == "none" or pos_enc == "NONE":
            self.position_embeddings = None
        else:
            raise ValueError("pos_enc canonly be: BERT")

        # Setting up the conditional embeddings; TODO: can this fit into the fg/fe framework instead?
        self.conditional_embeddings = nn.ModuleDict()
        if conditional_input_types is not None:
            for name, spec in conditional_input_types.items():
                if spec["type"] == "learned":
                    self.conditional_embeddings[name] = nn.Embedding(spec["vocab_size"], d_model)
                elif spec["type"] == "predefined":
                    self.conditional_embeddings[name] = None  # no need to specify anything, loads in directly
                else:
                    raise ValueError(f"conditional_input_types.{name}['type'] must be either 'learned' or 'predefined'")

        encoder_layer = instantiate_from_config(encoder_layer_parameters)
        self.transformer_encoder = instantiate_from_config(encoder_parameters, encoder_layer)

        # Initialize the [CLS] token as a learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, inputs, conditional_tokens=None, attention_mask=None):
        """LM model.

        Args:
            inputs: This is either integers if IDs or bf16/fp32
                floats for predefined embeddings
            conditional_tokens: _description_. Defaults
                to None.
            attention_mask: A tensor of shape [batchsize, seqlen] where 1/True
                represents no attention and 0/False represents that attention should be used

        Returns:
            torch.tensor: The predicted outputs before cross entropy loss.

        """
        identity_inputs, expression_inputs = inputs

        input_embeds = self.fc.embed_cells(
            identity_inputs,
            self.gene_embeddings,
            expression_inputs,
            self.expression_embeddings,
        )

        batch_size = identity_inputs.size(0)
        seq_length = input_embeds.size(1)

        # Positional Encoding
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_embeds.device,
        ).expand((batch_size, -1))

        if self.position_embeddings is not None:
            input_embeds += self.position_embeddings(position_ids)

        # Dynamically adding the conditional tokens, if there are any
        if conditional_tokens is not None:
            assert isinstance(
                conditional_tokens,
                dict,
            ), "conditional_tokens must be a dictionary of names and IDs or embeddings to add to the input"
            assert (
                len(self.conditional_embeddings) > 0
            ), "This was not initialized properly, there are no conditional embeddings to add to the input"
            for name, embed in self.conditional_embeddings.items():
                if embed is not None:
                    input_embeds += embed(conditional_tokens[name])
                else:
                    input_embeds += conditional_tokens[name]
        else:
            assert len(self.conditional_embeddings) == 0, (
                "This model was initialized with conditional tokens, but none were passed in the forward pass. "
                "Please pass in the conditional tokens"
            )

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
        transformer_encoder_output = self.transformer_encoder(input_embeds, src_key_padding_mask=attention_mask)
        return transformer_encoder_output


class CellPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        cls_emb = encoder_output[:, 0, :]
        logits = self.decoder(cls_emb.unsqueeze(1)).squeeze(1)
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=encoder_output,
            cls_embeddings=cls_emb,
        )


class SeqPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        cls_emb = encoder_output[:, 0, :]
        logits = self.decoder(encoder_output[:, 1:, :])
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=encoder_output,
            cls_embeddings=cls_emb,
        )


class ExpressionPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        logits = self.decoder(encoder_output)
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=logits,
            cls_embeddings=logits,
        )


class LinearDecoderMixin(nn.Module):
    def __init__(self, dim_in: int, dim_out: Optional[int] = None, dropout: float = 0.0, **kwargs):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.decoder = nn.Sequential(
            nn.Linear(dim_in, dim_out, **kwargs),
            nn.Dropout(dropout),
        )


class FFN(nn.Module):
    def __init__(self, dim_in: int, dim_out: Optional[int] = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()

        dim_inner = int(dim_in * mult)
        if dim_out is None:
            dim_out = dim_in

        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class PreNormResidual(nn.Module):
    def __init__(self, module: nn.Module, dim: int):
        super().__init__()
        self.mod = module
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        res = self.mod(self.norm(x))
        assert res.shape == x.shape, "Input and output size must be the same for residual operations"
        return res + x


class LinearCellPredHead(CellPredHeadMixin, LinearDecoderMixin):
    """Linear cell prediction head."""


class ExpressionOnlyCellPredHead(ExpressionPredHeadMixin, LinearDecoderMixin):
    """Logistic regression prediction head.

    Put expression be the input

    """


class LinearSeqPredHead(SeqPredHeadMixin, LinearDecoderMixin):
    """Linear sequence prediction head."""


class Reducer(nn.Module, ABC):
    """Reduce a list of `n` tensors into a single tensor.

    Each tensor in the list must have dimensionality `(batch_size, dim_in)`. The
    reduction may be symmetric or asymmetric.

    """

    def __init__(self, dim_in: int):
        super().__init__()
        self.dim_in = dim_in

    @abstractmethod
    def forward(self, tensors: list[Tensor]): ...


class SumReducer(Reducer):
    def forward(self, tensors: list[Tensor]):
        return torch.sum(torch.stack(tensors, axis=0), axis=0)


class MeanReducer(Reducer):
    def forward(self, tensors: list[Tensor]):
        return torch.mean(torch.stack(tensors, axis=0), axis=0)


class AsymmetricConcatReducer(Reducer):
    def __init__(self, dim_in: int):
        super().__init__(dim_in=dim_in)
        self.pair_embedder = nn.Linear(2 * dim_in, dim_in)

    def forward(self, tensors: list[Tensor]):
        concatenated = torch.cat(tensors, dim=-1)
        return self.pair_embedder(concatenated)


class SymmetricConcatReducer(Reducer):
    def __init__(self, dim_in: int):
        super().__init__(dim_in=dim_in)
        self.pair_embedder = nn.Linear(2 * dim_in, dim_in)

    def forward(self, tensors: list[Tensor]):
        concatenated_1 = torch.cat(tensors, dim=2)
        concatenated_2 = torch.cat(list(reversed(tensors)), dim=2)

        encoded = self.pair_embedder(concatenated_1) + self.pair_embedder(concatenated_2)
        return encoded
