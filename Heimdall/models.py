### Imports and Helper Functions
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.utils.data import Dataset
from random import randrange, random
from torch.nn import CrossEntropyLoss
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
## Flash attention Imports
from transformers import BertConfig
from dataclasses import dataclass, field


try:
    from flash_attn.models.bert import BertEncoder
    print("FlashAttention Library Successfully Loaded")
except ImportError:
    print("Warning: FlashAttention Not Installed, when initializing model make sure to use default Transformers")


import torch
import torch.nn as nn


from dataclasses import dataclass, field
from typing import Union, Dict, Optional



####
# Heimdall blackbox model
####

@dataclass
class TransformerConfig:
    vocab_size: int = 1000
    prediction_dim: int = 2
    d_model: int = 128
    nhead: int = 2
    num_encoder_layers: int = 2
    max_seq_length: int = 1000
    pos_enc: str = "BERT"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12


class Heimdall_Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, input_type: str, conditional_input_types: Optional[dict] = None):
        super(Heimdall_Transformer, self).__init__()
        """
        - The config is primarily the transformer config

        - input_type: 'learned'or 'predefined'

        - conditional_input_types: {
            'binned_gene_expression_embeddings' : {
                'type': 'learned',
                'vocab_size': 512,
                }
                
            'ESM_embeddings' : {
                'type': 'predefined',
                'vocab_size': -1
                }
        }
        """
        self.config = config
        self.conditional_input_types = conditional_input_types
        self.input_type = input_type

        # Set up the Input Embedding layers
        if input_type == 'learned':
            self.input_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        elif input_type == 'predefined':
            pass
        else:
            raise ValueError("input_type must be either 'learned' or 'predefined'")

        # Setting up explicit Positional Encodings
        if config.pos_enc == "BERT":
            self.position_embeddings = nn.Embedding(config.max_seq_length + 1, config.d_model) ## +1 cuz of CLS
        elif config.pos_enc == "sincos":
            raise NotImplementedError("Sine-Cosine Positional Encodings are not implemented yet")
        else:
            raise ValueError("config.pos_enc canonly be: BERT")

        ## Setting up the conditional embeddings
        self.conditional_embeddings = {}
        if conditional_input_types is not None:
            for name, spec in conditional_input_types.items():
                if spec['type'] == 'learned':
                    self.conditional_embeddings[name] = nn.Embedding(spec['vocab_size'], config.d_model)
                elif spec['type'] == 'predefined':
                    self.conditional_embeddings[name] = None ## no need to specify anything, loads in directly
                else:
                    raise ValueError(f"conditional_input_types.{name}['type'] must be either 'learned' or 'predefined'")

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True  # BERT uses LayerNorm before self-attention and feedforward networks
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.decoder = nn.Linear(config.d_model, config.prediction_dim, bias=True)


        # Initialize the [CLS] token as a learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))


    def forward(self, inputs, conditional_tokens = None, attention_mask = None):
        """
        Args:
            inputs (torch tensor): this is either integers if IDs or bf16/fp32 floats for predefined embeddings
            conditional_tokens (dictionary, optional): _description_. Defaults to None.
            attention_mask (Attention Mask for Padding, optional): NOT IMPLEMENTED YET. Defaults to None.
        Returns:
            torch tensor: the predicted outputs before cross entropy loss
        """

        # Embedding layer
        if self.input_type == 'learned':
            input_embeds = self.input_embeddings(inputs)
        elif self.input_type == 'predefined':
            input_embeds = inputs
        else:
            raise ValueError("input_type must be either 'learned' or 'predefined'")


        # Concatenate [CLS] token to the beginning of every sequence in the batch
        cls_tokens = self.cls_token.expand(inputs.size(0), -1, -1)  # Expand to match batch size

        ## Positional Encoding
        seq_length = input_embeds.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeds.device).expand((inputs.size(0), -1))
        input_embeds += self.position_embeddings(position_ids)

        ## Dynamically adding the conditional tokens, if there are any
        if conditional_tokens is not None:
            assert type(conditional_tokens) == dict, "conditional_tokens must be a dictionary of names and IDs or embeddings to add to the input"
            assert len(self.conditional_embeddings) > 0, "This was not initialized properly, there are no conditional embeddings to add to the input"
            for name, embed in self.conditional_embeddings.items():
                if embed is not None:
                    # print(conditional_tokens[name])
                    input_embeds += embed(conditional_tokens[name])
                else:
                    input_embeds += conditional_tokens[name]
        else:
            assert len(self.conditional_embeddings) == 0, "This model was initialized with conditional tokens, but none were passed in the forward pass. Please pass in the conditional tokens"


        ## Add the CLS Token
        input_embeds = torch.cat([cls_tokens, input_embeds], dim=1)

        # Encoder
        encoder_output = self.encoder(input_embeds, src_key_padding_mask=attention_mask)

        ## Taking just the CLS token to pass to the decoder
        CLS_token = encoder_output[:, 0, :]

        # Decoder
        prediction_scores = self.decoder(CLS_token)

        return prediction_scores






### junk below, for reference for when Flash Attention is set uplater

####
# better performing TransformerMLM
####

# @dataclass
# class TransformerConfig:
#     vocab_size: int = 12
#     d_model: int = 512
#     num_attention_heads: int = 4
#     layer_count: int = 6
#     max_length: int = 8192
#     conv_forward_type: str = "simple"
#     hidden_dropout_prob: float = 0.1
#     attention_probs_dropout_prob: float = 0.1
#     conv_downsample: bool = True
#     kernel_size: int = 16


# class TransformerMLM(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         layer_count = config.layer_count
#         hidden_features = config.d_model
#         max_length = config.max_length
#         conv_forward_type = config.conv_forward_type
#         conv_downsample = config.conv_downsample
#         kernel_size = config.kernel_size
#         num_attention_heads = config.num_attention_heads
#         hidden_dropout_prob = config.hidden_dropout_prob
#         attention_probs_dropout_prob = config.attention_probs_dropout_prob
#         vocab_size = config.vocab_size
#         self.config = config

#         self.conv_downsample = conv_downsample ## whether or not to downsample the input to a certain size

#         if self.conv_downsample == True:
#             self.convhead = ConvHead(hidden_size=hidden_features, kernel_size = kernel_size, forward_type=conv_forward_type)

#         transformer_config = BertConfig(hidden_size=hidden_features, 
#                                         num_hidden_layers=layer_count, 
#                                         num_attention_heads=num_attention_heads, 
#                                         intermediate_size=hidden_features * 4, 
#                                         hidden_act="gelu", 
#                                         hidden_dropout_prob=hidden_dropout_prob, 
#                                         attention_probs_dropout_prob=attention_probs_dropout_prob, 
#                                         max_position_embeddings=max_length,
#                                         use_flash_attn=True, ### use this to toggle between flash attention and not
#                                         position_embedding_type='rotary', ## if use_flash_attn true, then 'rotary' is needed for Positional Encoding
#                                         rotary_emb_dim= (hidden_features // num_attention_heads), 
#                                         kernel_size = -1, ## arbitrary
#                                         vocab_size = -1 ## one-hot of size 5, ATCG and Mask
#                                         )
#         # Use the default transformer implementation, provided transformer_config is passed correctly
#         self.encoder = BertEncoder(transformer_config)  # Assuming BertEncoder accepts a config
#         self.embed = nn.Embedding(vocab_size, hidden_features)
#         self.lm_head = nn.Linear(hidden_features, vocab_size, bias=False)
#         self.lm_head.weight = self.embed.weight  # Tie weights

#     def backbone(self, input_ids, return_all_hidden = False, return_all_hidden_device='cpu'):
#         x = self.embed(input_ids)
#         if self.conv_downsample == True:
#             x = self.convhead(x)
#         # print(f"x.dtype: {x.dtype}")
#         # print(f"weights.dtype: {self.embed.weight.dtype}")

#         x, x_all_layers = self.encoder(x, return_all_hidden = return_all_hidden, return_all_hidden_device=return_all_hidden_device)
#         return x, x_all_layers


#     def forward(self, input_ids): ## out_type \in [recon, pred]
#         x, _ = self.backbone(input_ids)
#         out = self.lm_head(x)
#         return out, None

