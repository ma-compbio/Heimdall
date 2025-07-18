from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import anndata as ad
import awkward as ak
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.nn import Module

from Heimdall.fe import Fe
from Heimdall.fg import Fg


class Fc(ABC):
    """Abstraction for cell embedding.

    Args:
        fg: `Fg` used for this `Fc` implementation.
        fe: `Fe` used for this `Fe` implementation.
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        max_input_length: maximum number of identity/expression tokens to consider for each cell.
            Extra tokens are limited.

    """

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        embedding_parameters: DictConfig,
        num_metadata_tokens: int = 0,
        max_input_length: Optional[int] = None,
        float_dtype: str = "float32",
    ):
        self.fg = fg
        self.fe = fe
        self.adata = adata
        self.max_input_length = max_input_length
        self.float_dtype = float_dtype
        self.embedding_parameters = OmegaConf.to_container(embedding_parameters, resolve=True)

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Retrieve `identity_inputs`, `expression_inputs` and `padding_mask`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """
        identity_indices, expression_inputs = self.fe[cell_index]

        gene_list = self.adata.var_names[identity_indices]  # convert to ENSEMBL Gene Names
        identity_inputs = self.fg[gene_list]  # convert the genes into fg

        if len(identity_inputs) != len(expression_inputs):
            raise ValueError(
                "Gene identity and expression inputs do not have the same shape; `Fg` and `Fe` are incompatible.",
            )

        # Padding and truncating
        # breakpoint()
        identity_inputs, expression_inputs = self.tailor(
            identity_inputs,
            expression_inputs,
        )

        padding_mask = expression_inputs == self.fe.pad_value
        return identity_inputs, expression_inputs, padding_mask

    def pad(self, cell_tokenization: ak.Array) -> tuple[ak.Array, ak.Array]:
        """Pad tokenization that is smaller than desired input length.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                dof a cell.

        """

        _, input_length = cell_tokenization.shape
        pad_widths = ((0, 0), (0, self.max_input_length - input_length))
        padded = np.pad(
            cell_tokenization.astype(self.float_dtype),
            pad_widths,
            "constant",
            constant_values=(0, np.nan),
        )

        padded[0, np.isnan(padded[0]).nonzero()] = self.fg.pad_value
        padded[1, np.isnan(padded[1]).nonzero()] = self.fe.pad_value

        return padded

    @abstractmethod
    def limit(self, cell_tokenization: NDArray) -> NDArray:
        """Limit tokenization that exceeds the desired input length.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                of a cell.

        """

    def tailor(
        self,
        gene_tokenization,
        expression_tokenization,
    ) -> NDArray | ak.Array:

        # first, drop any NaN values here
        # Assuming gene_tokenization is a pandas Series and expression_tokenization is a numpy array
        valid_mask = ~np.isnan(expression_tokenization)

        filtered_gene_tokenization = gene_tokenization[valid_mask]
        filtered_expression_tokenization = expression_tokenization[valid_mask]

        cell_tokenization = np.stack([filtered_gene_tokenization.values, filtered_expression_tokenization], axis=0)
        # cell_tokenization = np.stack([gene_tokenization, expression_tokenization], axis=0)

        _, input_length = cell_tokenization.shape

        if input_length > self.max_input_length:
            return self.limit(cell_tokenization)
        return self.pad(cell_tokenization)

    @abstractmethod
    def embed_cells(
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

        Returns:
            Embeddings of cells.

        """


class GeneformerFc(Fc):
    """Implementation of Geneformer cell embedding."""

    def limit(self, cell_tokenization: NDArray) -> NDArray:
        return cell_tokenization[:, : self.max_input_length]

    def embed_cells(
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


class DummyFc(Fc):
    """Dummy `Fc` that does not tailor the size of the input."""

    def tailor(
        self,
        gene_tokenization,
        expression_tokenization,
    ) -> NDArray | ak.Array:

        cell_tokenization = np.stack([gene_tokenization, expression_tokenization], axis=0)
        _, input_length = cell_tokenization.shape

        return cell_tokenization

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Dummy `__getitem__` for model that does not need an `Fc`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """
        identity_indices, expression_inputs = self.fe[cell_index]
        padding_mask = np.zeros(self.max_input_length)

        return identity_indices, expression_inputs, padding_mask

    def limit(self, cell_tokenization: NDArray) -> NDArray:
        pass

    def embed_cells(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:

        pass


class ScGPTFc(Fc):
    """Implementation of scGPT cell embedding."""

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        embedding_parameters: OmegaConf,
        **fc_kwargs,
    ):
        super().__init__(fg, fe, adata, embedding_parameters, **fc_kwargs)
        seed = 0  # TODO: make this configurable???
        self.rng = np.random.default_rng(seed)

    def limit(self, cell_tokenization: NDArray) -> NDArray:
        # Shape: (2, N)
        expression_values = cell_tokenization[1]

        # Separate indices
        nonzero_indices = np.where(expression_values != 0)[0]
        zero_indices = np.where(expression_values == 0)[0]

        # First: sample nonzero expression tokens
        num_nonzero_to_sample = min(len(nonzero_indices), self.max_input_length)
        selected_nonzero = self.rng.choice(nonzero_indices, num_nonzero_to_sample, replace=False)

        # If needed: sample zero-expression tokens to fill up
        num_remaining = self.max_input_length - num_nonzero_to_sample
        if num_remaining > 0:
            selected_zero = self.rng.choice(zero_indices, num_remaining, replace=False)
            final_indices = np.concatenate([selected_nonzero, selected_zero])
        else:
            final_indices = selected_nonzero

        # Optionally shuffle to avoid position bias, but we dont need to because the gene ids are the position
        # self.rng.shuffle(final_indices)

        return cell_tokenization[:, final_indices]

    def embed_cells(
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


class UCEFc(Fc):
    """Chromosome-aware implementation of cell embedding."""

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        embedding_parameters: OmegaConf,
        gene_metadata_filepath: str | Path,
        ensembl_dir: str | Path,
        species: str,
        # chroms: Optional[NDArray] = None,
        # starts: Optional[NDArray] = None,
        **fc_kwargs,
    ):
        """
        Args:
            chroms: Chromosome IDs for each gene.
            starts: Genomic start positions of genes on their chromosomes.
        """
        super().__init__(fg, fe, adata, embedding_parameters, **fc_kwargs)
        seed = 0  # TODO: make this configurable???
        self.rng = np.random.default_rng(seed)

        # https://github.com/snap-stanford/UCE/blob/8227a65cdd021b9186ef86671d2aef5c895c8e4b/data_proc/data_utils.py#L155
        # TODO: load chromosome one-hot encoding and start positions for all genes

        self.gene_metadata = pd.read_csv(gene_metadata_filepath)
        self.ensembl_dir = ensembl_dir
        self.species = species

        self.gene_metadata["spec_chrom"] = pd.Categorical(
            self.gene_metadata["species"] + "_" + self.gene_metadata["chromosome"],
        )

        spec_chrom = self.gene_metadata[self.gene_metadata["species"] == self.species].set_index("gene_symbol")

        # symbol_to_ensembl_mapping = symbol_to_ensembl_from_ensembl(
        #     data_dir=self.ensembl_dir,
        #     genes=spec_chrom.index.tolist(),
        #     species=self.species,
        # )
        # spec_chrom.index = spec_chrom.index.map(symbol_to_ensembl_mapping.mapping_reduced)

        try:
            # NOTE: below is different from UCE...
            gene_names = [k.upper() for k in self.adata.var["gene_symbol"]]
            # gene_chrom = spec_chrom.loc[gene_names]
            gene_chrom = spec_chrom.reindex(gene_names, copy=True)
        except KeyError as e:
            raise ValueError(
                "Input AnnData cannot contain gene names that are unmapped in the chromosome metadata.",
            ) from e

        # TODO: for pretraining, we should keep extraneous codes (i.e. no `remove_unused_categories()`)
        dataset_chroms = gene_chrom["spec_chrom"].cat.remove_unused_categories().cat.codes
        print("Max Code:", max(dataset_chroms))
        dataset_pos = gene_chrom["start"].values

        self.unique_chromosomes = np.unique(dataset_chroms)

        self.chroms = dataset_chroms
        self.starts = dataset_pos

        self.chrom_token_offset = 1

    def limit(self, cell_tokenization: NDArray) -> NDArray:
        return cell_tokenization[:, : self.max_input_length]

    def tailor(
        self,
        gene_tokenization,
        expression_tokenization,
    ) -> NDArray | ak.Array:

        # first, drop any NaN values here
        # Assuming gene_tokenization is a pandas Series and expression_tokenization is a numpy array
        valid_mask = ~np.isnan(expression_tokenization)

        gene_tokenization = gene_tokenization[valid_mask].to_numpy()
        expression_tokenization = expression_tokenization[valid_mask]

        choosen_chrom = self.chroms.iloc[gene_tokenization]

        chrom_sort = np.argsort(choosen_chrom)

        gene_tokenization = gene_tokenization[chrom_sort]
        expression_tokenization = expression_tokenization[chrom_sort]

        new_chrom = self.chroms.iloc[gene_tokenization]
        choosen_starts = self.starts[gene_tokenization]

        unique_chromosomes = np.unique(new_chrom)
        self.rng.shuffle(unique_chromosomes)

        num_chromosomes = len(unique_chromosomes)
        raw_sequence_length = len(new_chrom) + 2 * num_chromosomes

        grouped_gene_tokenization = np.full(raw_sequence_length, self.fg.pad_value)
        grouped_expression_tokenization = np.full(raw_sequence_length, self.fe.pad_value)

        sequence_index = 0

        for chromosome in unique_chromosomes:
            placeholder_id = -(chromosome + self.chrom_token_offset + 1)
            grouped_gene_tokenization[sequence_index] = placeholder_id
            grouped_expression_tokenization[sequence_index] = placeholder_id
            # ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET
            # token of this chromosome # i = 1 next token is a chrom open

            sequence_index += 1
            # now sort the genes by start order within the chroms
            (chromosome_index,) = np.where(new_chrom == chromosome)
            num_chromosome_genes = len(chromosome_index)

            sort_by_start = np.argsort(
                choosen_starts[chromosome_index],
            )  # start chromosome_indexations for this chromsome

            chromosome_genes = gene_tokenization[chromosome_index[sort_by_start]]
            chromosome_expression = expression_tokenization[chromosome_index[sort_by_start]]

            grouped_gene_tokenization[sequence_index : (sequence_index + num_chromosome_genes)] = chromosome_genes
            grouped_expression_tokenization[sequence_index : (sequence_index + num_chromosome_genes)] = (
                chromosome_expression
            )

            sequence_index += num_chromosome_genes

            grouped_gene_tokenization[sequence_index] = -self.chrom_token_offset
            grouped_expression_tokenization[sequence_index] = -self.chrom_token_offset
            # ordered_choice_idx[i] = args.chrom_token_right_idx # add the chrom sep again

            sequence_index += 1  # add the closing token again

        cell_tokenization = np.stack([grouped_gene_tokenization, grouped_expression_tokenization], axis=0)
        # cell_tokenization = np.stack([gene_tokenization, expression_tokenization], axis=0)

        _, input_length = cell_tokenization.shape

        if input_length > self.max_input_length:
            return self.limit(cell_tokenization)
        return self.pad(cell_tokenization)

    def embed_cells(
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
        chrom_token_indices = -chrom_token_indices - self.chrom_token_offset

        identity_inputs[chrom_token_mask] = 0

        gene_embeddings = gene_embedding_layer(identity_inputs)
        # TODO: want to use bins with this, but currently ignoring
        # print(chrom_token_indices.cpu())

        gene_embeddings[chrom_token_mask] = metadata_embedding_layer(chrom_token_indices)

        return gene_embeddings


class ScBERTFc(ScGPTFc):
    """Implementation of scBERT cell embedding."""

    # TODO: is ScBERTFc actually the same as ScGPTFc?
