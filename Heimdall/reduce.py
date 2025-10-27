from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module

from Heimdall.fc import Fc


class Reduce(ABC):
    def __init__(
        self,
        fc: Fc,
    ):
        self.fc = fc

    @abstractmethod
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cell batch using the embedding layers.

        It can be assumed that both the identity inputs and the expression inputs have been padded/
        limited at this stage, i.e. they are regular-shaped tensors.

        Args:
            identity_inputs: batched gene identity inputs
            gene_embedding_layer: Torch module for embedding based on gene identity.
            expression_inputs: batched gene expression inputs
            expression_embedding_layer: Torch module for embedding based on expression.
            metadata_embedding_layer: Torch module for embedding based on metadata.

        Returns:
            Embeddings of cells.

        """


class IdentityReduce(Reduce):
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """Geneformer cell embedding function.

        Ignores expression embedding layer; uses embeddings based on identity embeddings.

        Args:
            gene_embedding_layer:  # TODO: fill out
            expression_embedding_layer: # TODO fill out

        """

        embeddings = gene_embedding_layer(identity_inputs)
        return embeddings


class SumReduce(Reduce):
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """ScGPT cell embedding callback.

        TODO: add "conditional tokens" (see Methods of https://www.nature.com/articles/s41592-024-02201-0#Sec14)

        Args:
            gene_embedding_layer:  # TODO: fill out
            expression_embedding_layer: # TODO fill out

        """
        # Convert str float_dtype -> actual torch dtype
        # torch_dtype = getattr(torch, self.float_dtype)
        # Cast expression_inputs to float_dtype
        expression_inputs = expression_inputs.to(torch.float32)

        gene_embeddings = gene_embedding_layer(identity_inputs)
        expression_embeddings = expression_embedding_layer(expression_inputs)

        return gene_embeddings + expression_embeddings


class ChromosomeReduce(Reduce):
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cells using chromosome-aware sequences."""

        chrom_token_mask = identity_inputs < 0
        chrom_token_indices = identity_inputs[identity_inputs < 0]
        chrom_token_indices = -chrom_token_indices - self.fc.chrom_token_offset

        identity_inputs[chrom_token_mask] = 0

        gene_embeddings = gene_embedding_layer(identity_inputs)

        gene_embeddings[chrom_token_mask] = metadata_embedding_layer(chrom_token_indices)

        return gene_embeddings


class ChromosomeSumReduce(Reduce):
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cells using chromosome-aware sequences."""

        chrom_token_mask = identity_inputs < 0
        chrom_token_indices = identity_inputs[identity_inputs < 0]
        chrom_token_indices = -chrom_token_indices - self.fc.chrom_token_offset

        identity_inputs[chrom_token_mask] = 0
        expression_inputs[chrom_token_mask] = 0

        gene_embeddings = gene_embedding_layer(identity_inputs)
        expression_embeddings = expression_embedding_layer(expression_inputs)
        meta_emb = metadata_embedding_layer(chrom_token_indices)

        meta_emb_gene = meta_emb.to(dtype=gene_embeddings.dtype, device=gene_embeddings.device)
        meta_emb_expr = meta_emb.to(dtype=expression_embeddings.dtype, device=expression_embeddings.device)

        gene_embeddings[chrom_token_mask] = meta_emb_gene
        expression_embeddings[chrom_token_mask] = meta_emb_expr

        return gene_embeddings + expression_embeddings
