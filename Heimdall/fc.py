from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from Heimdall.fe import Fe
from Heimdall.fg import Fg
from Heimdall.utils import instantiate_from_config


class Fc:
    """Abstraction for cell embedding.

    Args:
        fg: `Fg` used for this `Fc` implementation.
        fe: `Fe` used for this `Fc` implementation.
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        max_input_length: maximum number of identity/expression tokens to consider for each cell.
            Extra tokens are limited.

    """

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        tailor_config: DictConfig,
        order_config: DictConfig,
        reduce_config: DictConfig,
        embedding_parameters: DictConfig,
        max_input_length: Optional[int] = None,
        float_dtype: str = "float32",
        rng: int | np.random.Generator = 0,
    ):
        self.fg = fg
        self.fe = fe
        self._adata = adata
        self.max_input_length = max_input_length
        self.float_dtype = float_dtype
        self.embedding_parameters = OmegaConf.to_container(embedding_parameters, resolve=True)
        self.rng = np.random.default_rng(rng)

        self.tailor = instantiate_from_config(tailor_config, fc=self)
        self.order = instantiate_from_config(order_config, fc=self)
        self.reduce = instantiate_from_config(reduce_config, fc=self)

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Retrieve `identity_inputs`, `expression_inputs` and `padding_mask`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """

        if cell_index == -1:  # Dummy `cell_index`
            identity_inputs = pd.array(np.full(self.max_input_length, self.fg.pad_value), dtype="Int64")
            expression_inputs = np.full(self.max_input_length, self.fe.pad_value)
        else:
            identity_indices, expression_inputs = self.fe[cell_index]

            gene_list = self.adata.var_names[identity_indices]  # convert to ENSEMBL Gene Names
            identity_inputs = self.fg[gene_list]  # convert the genes into fg

        if len(identity_inputs) != len(expression_inputs):
            raise ValueError(
                "Gene identity and expression inputs do not have the same shape; `Fg` and `Fe` are incompatible.",
            )

        # first, drop any `NaN` values here
        # Assuming gene_tokenization is a pandas IntegerArray and expression_tokenization is a numpy array
        # TODO: what does `NaN` represent here?
        valid_mask = ~np.isnan(expression_inputs)

        identity_inputs = identity_inputs[valid_mask].to_numpy()
        expression_inputs = expression_inputs[valid_mask]

        gene_order = self.order(identity_inputs, expression_inputs)

        # Padding and truncating
        identity_inputs, expression_inputs = self.tailor(
            identity_inputs,
            expression_inputs,
            gene_order,
        )
        padding_mask = expression_inputs == self.fe.pad_value
        return identity_inputs, expression_inputs, padding_mask

    @property
    def adata(self):
        return self._adata

    @adata.setter
    def adata(self, val):
        self._adata = val
        self.fg.adata = val
        self.fe.adata = val


class ChromosomeAwareFc(Fc):
    def __init__(
        self,
        *fc_args,
        gene_metadata_filepath: str | Path,
        ensembl_dir: str | Path,
        species: str,
        **fc_kwargs,
    ):
        """
        Args:
            gene_metadata_filepath: path to gene metadata .csv
            ensembl_dir: path to directory in which Ensembl mapping file is stored
            species: species from which single-cell dataset is derived
        """

        super().__init__(*fc_args, **fc_kwargs)

        self.gene_metadata = pd.read_csv(gene_metadata_filepath)
        self.ensembl_dir = ensembl_dir
        self.species = species

        self.gene_metadata["spec_chrom"] = pd.Categorical(
            self.gene_metadata["species"] + "_" + self.gene_metadata["chromosome"],
        )

        # https://github.com/snap-stanford/UCE/blob/8227a65cdd021b9186ef86671d2aef5c895c8e4b/data_proc/data_utils.py#L155
        # TODO: load chromosome one-hot encoding and start positions for all genes

        # symbol_to_ensembl_mapping = symbol_to_ensembl_from_ensembl(
        #     data_dir=self.ensembl_dir,
        #     genes=spec_chrom.index.tolist(),
        #     species=self.species,
        # )
        # spec_chrom.index = spec_chrom.index.map(symbol_to_ensembl_mapping.mapping_reduced)
        self.extract_gene_positions()
        self.chrom_token_offset = 1

    def extract_gene_positions(self):
        spec_chrom = self.gene_metadata[self.gene_metadata["species"] == self.species].set_index("gene_symbol")
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

    @Fc.adata.setter
    def adata(self, val):
        Fc.adata.fset(self, val)
        self.extract_gene_positions()


class DummyFc(Fc):
    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        tailor_config: DictConfig,
        order_config: DictConfig,
        reduce_config: DictConfig,
        embedding_parameters: DictConfig,
        max_input_length: Optional[int] = None,
        float_dtype: str = "float32",
        rng: int | np.random.Generator = 0,
    ):
        self.fg = fg
        self.fe = fe
        self.adata = adata
        self.max_input_length = max_input_length

    """Dummy `Fc` that does not tailor the size of the input."""

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Dummy `__getitem__` for model that does not need an `Fc`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """
        identity_indices, expression_inputs = self.fe[cell_index]
        padding_mask = np.zeros(self.max_input_length)

        return identity_indices, expression_inputs, padding_mask
