"""Heimdall model."""

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from Heimdall.cell_representations import CellRepresentation
from Heimdall.utils import instantiate_from_config

# try:
#     from flash_attn.models.bert import BertEncoder
#
#     print("FlashAttention Library Successfully Loaded")
# except ImportError:
#     print("Warning: FlashAttention Not Installed, when initializing model make sure to use default Transformers")


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


class HeimdallTransformer(nn.Module):
    def __init__(
        self,
        data: CellRepresentation,
        config: DictConfig,
        input_type: str,
        conditional_input_types: Optional[dict] = None,
        embedding_layer=None,
    ):
        super().__init__()
        """Heimdall transformer model.

        Args:
            data: Cell representation data object.
            config: The transformer config.
            input_type: "learned" or "predefined"
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
        self.config = config
        self.conditional_input_types = conditional_input_types
        self.input_type = input_type

        self.embedding_layer = embedding_layer
        # Set up the Input Embedding layers
        if self.embedding_layer is not None:
            self.input_embeddings = self.embedding_layer
            print(f"Using provided pretrained embedding layer with shape: {embedding_layer.weight.shape}")

        self.num_labels = data.num_tasks
        self.vocab_size = data.sequence_length + 2  # <PAD> and <MASK> TODO: data.vocab_size
        self.max_seq_length = data.sequence_length

        # Set up the Input Embedding layers
        if input_type == "learned":
            self.input_embeddings = nn.Embedding(self.vocab_size, config.d_model)
        elif input_type == "predefined":
            pass
        else:
            raise ValueError("input_type must be either 'learned' or 'predefined'")

        # Setting up explicit Positional Encodings
        if config.pos_enc == "BERT":
            self.position_embeddings = nn.Embedding(self.max_seq_length + 1, config.d_model)  # +1 cuz of CLS
        elif config.pos_enc == "sincos":
            raise NotImplementedError("Sine-Cosine Positional Encodings are not implemented yet")
        else:
            raise ValueError("config.pos_enc canonly be: BERT")

        # Setting up the conditional embeddings
        self.conditional_embeddings = nn.ModuleDict()
        if conditional_input_types is not None:
            for name, spec in conditional_input_types.items():
                if spec["type"] == "learned":
                    self.conditional_embeddings[name] = nn.Embedding(spec["vocab_size"], config.d_model)
                elif spec["type"] == "predefined":
                    self.conditional_embeddings[name] = None  # no need to specify anything, loads in directly
                else:
                    raise ValueError(f"conditional_input_types.{name}['type'] must be either 'learned' or 'predefined'")

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True,
            norm_first=True,  # BERT uses LayerNorm before self-attention and feedforward networks
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.head = instantiate_from_config(config.head_config, dim_in=config.d_model, dim_out=self.num_labels)

        # Initialize the [CLS] token as a learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

    def forward(self, inputs, labels=None, conditional_tokens=None, attention_mask=None):
        """Forward function.

        Args:
            inputs: Inputs.
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression
                loss. Indices should be in `[0, ..., config.num_labels - 1]`.
                If `config.num_labels == 1` a regression loss is computed
                (Mean-Square loss), If `config.num_labels > 1` a classification
                loss is computed (Cross-Entropy).

        """
        # handling when tehre are no conditional tokens supplied
        if conditional_tokens is not None and len(conditional_tokens) == 0:
            conditional_tokens = None

        if isinstance(inputs, list):
            # TODO: replace with proper handling
            warnings.warn(
                "Paired input model not setup corectly yet, only use for dev",
                UserWarning,
                stacklevel=2,
            )
            outputs1, outputs2 = (self.lm_model(inputs[i], conditional_tokens, attention_mask) for i in range(2))
            outputs = TransformerOutput(
                **{key: getattr(outputs1, key) + getattr(outputs2, key) for key in outputs1.__dict__},
            )
        else:
            outputs = self.lm_model(inputs, conditional_tokens, attention_mask)

        return outputs

        # loss = None
        # if labels is not None:
        #     labels = labels.to(logits.device)

        #     ## instantiating the problem type if it is not specified
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     ## obtaining the loss
        #     if self.config.problem_type == "regression":
        #         if self.use_huberloss:
        #             loss_fct = HuberLoss()
        #         else:
        #             loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     elif self.config.problem_type == "multi_label_classification":
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(logits, labels)

        # payload = {
        #     "loss" : loss,
        #     "logits" : logits
        # }

        # return payload

    def lm_model(self, inputs, conditional_tokens=None, attention_mask=None):
        """LM model.

        Args:
            inputs (torch tensor): This is either integers if IDs or bf16/fp32
                floats for predefined embeddings
            conditional_tokens (dictionary, optional): _description_. Defaults
                to None.
            attention_mask (Attention Mask for Padding, optional): NOT
                IMPLEMENTED YET. Defaults to None.

        Returns:
            torch.tensor: The predicted outputs before cross entropy loss.

        """

        # Embedding layer
        if self.input_type == "learned":
            input_embeds = self.input_embeddings(inputs)
        elif self.input_type == "predefined":
            input_embeds = inputs
        else:
            raise ValueError("input_type must be either 'learned' or 'predefined'")

        # Concatenate [CLS] token to the beginning of every sequence in the batch
        cls_tokens = self.cls_token.expand(inputs.size(0), -1, -1)  # Expand to match batch size

        # Positional Encoding
        seq_length = input_embeds.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_embeds.device,
        ).expand((inputs.size(0), -1))
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
                    # print(conditional_tokens[name])
                    input_embeds += embed(conditional_tokens[name])
                else:
                    input_embeds += conditional_tokens[name]
        else:
            assert len(self.conditional_embeddings) == 0, (
                "This model was initialized with conditional tokens, but none were passed in the forward pass. "
                "Please pass in the conditional tokens"
            )

        # Add the CLS Token
        input_embeds = torch.cat([cls_tokens, input_embeds], dim=1)

        # Encoder
        encoder_output = self.encoder(input_embeds, src_key_padding_mask=attention_mask)

        return self.head(encoder_output)


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
        logits = self.decoder(encoder_output[:, 1:, :])
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=encoder_output,
            cls_embeddings=encoder_output[:, 0, :],
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


class LinearSeqPredHead(SeqPredHeadMixin, LinearDecoderMixin):
    """Linear sequence prediction head."""
