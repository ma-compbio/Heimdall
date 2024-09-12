from abc import ABC, abstractmethod
from typing import Dict, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas.api.typing import NAType


class Fg(ABC):
    """Abstraction of the gene embedding mapping paradigm.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.

    """

    def __init__(self, adata: ad.AnnData, config: dict):
        self.adata = adata
        _, self.num_genes = adata.shape
        self.config = config

    @abstractmethod
    def preprocess_embeddings(self):
        """Preprocess gene embeddings and store them for use during model
        inference.

        Preprocessing may include anything from downloading gene embeddings from
        a URL to generating embeddings from scratch.

        """

    def __getitem__(self, gene_names: Sequence[str]) -> int | NAType:
        """Get the indices of genes in the embedding array.

        Args:
            gene_names: name of the gene as stored in `self.adata`.

        Returns:
            Index of gene in the embedding, or `pd.NA` if the gene has no mapping.

        """
        embedding_indices = self.adata.var.loc[gene_names, "identity_embedding_index"]
        if np.any(embedding_indices.isna()):
            raise KeyError(
                "At least one gene is not mapped in this Fg. Please remove such genes from consideration in the Fc.",
            )

        return embedding_indices


class PretrainedFg(Fg, ABC):
    """Abstraction for pretrained `Fg`s that can be loaded from disk.

    Raises:
        ValueError: if `config.d_embedding` is larger than embedding dimensionality given in filepath.

    """

    @abstractmethod
    def load_embeddings(self) -> Dict[str, NDArray]:
        """Load the embeddings from disk and process into map.

        Returns:
            A mapping from gene names to embedding vectors.

        """

    def preprocess_embeddings(self):
        embedding_map = self.load_embeddings()

        first_embedding = next(iter(embedding_map.values()))
        if len(first_embedding) < self.config.d_embedding:
            raise ValueError(
                f"Dimensionality of pretrained embeddings ({len(first_embedding)} is less than the embedding dimensionality specified in the config ({self.config.d_embedding}). Please decrease the embedding dimensionality to be compatible with the pretrained embeddings.",
            )

        valid_gene_names = list(embedding_map.keys())

        valid_mask = pd.array(np.isin(self.adata.var_names.values, valid_gene_names))
        num_mapped_genes = valid_mask.sum()
        (valid_indices,) = np.nonzero(valid_mask)

        index_map = valid_mask.astype(pd.Int64Dtype())
        index_map[~valid_mask] = None
        index_map[valid_indices] = np.arange(num_mapped_genes)

        self.adata.var["identity_embedding_index"] = index_map
        self.adata.var["identity_valid_mask"] = valid_mask

        self.gene_embeddings = np.zeros((num_mapped_genes, self.config.d_embedding), dtype=np.float64)

        for gene_name in self.adata.var_names:
            embedding_index = self.adata.var.loc[gene_name, "identity_embedding_index"]
            if not pd.isna(embedding_index):
                self.gene_embeddings[embedding_index] = embedding_map[gene_name][: self.config.d_embedding]

        print(f"Found {len(valid_indices)} genes with mappings out of {len(self.adata.var_names)} genes.")


class IdentityFg(Fg):
    """Identity mapping of gene names to embeddings.

    This is the simplest possible Fg; it implies the use of learnable gene
    embeddings that are initialized randomly, as opposed to the use of
    pretrained embeddings.

    """

    def preprocess_embeddings(self):
        self.gene_embeddings = None
        self.adata.var["embedding_index"] = np.arange(self.num_genes)
        self.adata.var["identity_valid_mask"] = np.full(self.num_genes, True)


class ESM2Fg(PretrainedFg):
    """Mapping of gene names to pretrained ESM2 embeddings."""

    def load_embeddings(self):
        raw_gene_embedding_map = torch.load(self.config.embedding_filepath)

        raw_gene_embedding_map = {
            gene_name: embedding.detach().cpu().numpy() for gene_name, embedding in raw_gene_embedding_map.items()
        }

        return raw_gene_embedding_map


class Gene2VecFg(PretrainedFg):
    """Mapping of gene names to pretrained Gene2Vec embeddings."""

    def load_embeddings(self):
        raw_gene_embedding_dataframe = pd.read_csv(self.config.embedding_filepath, sep=r"\s+", header=None, index_col=0)
        raw_gene_embedding_map = {
            gene_name: raw_gene_embedding_dataframe.loc[gene_name].values
            for gene_name in raw_gene_embedding_dataframe.index
        }

        return raw_gene_embedding_map
