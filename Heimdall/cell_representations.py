"""The Cell Representation Object for Processing."""

import pickle as pkl
import warnings
from functools import partial, wraps
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csc_array
from sklearn.utils import resample
from torch.utils.data import DataLoader, Subset

from Heimdall.datasets import Dataset
from Heimdall.fc import Fc
from Heimdall.fe import Fe
from Heimdall.fg import Fg
from Heimdall.utils import (
    get_cached_paths,
    get_value,
    heimdall_collate_fn,
    instantiate_from_config,
    symbol_to_ensembl_from_ensembl,
)


def check_states(
    meth: Optional[Callable] = None,
    *,
    adata: bool = False,
    processed_fcfg: bool = False,
    labels: bool = False,
    splits: bool = False,
):
    if meth is None:
        return partial(check_states, adata=adata, processed_fcfg=processed_fcfg)

    @wraps(meth)
    def bounded(self, *args, **kwargs):
        if adata:
            assert self.adata is not None, "no adata found, Make sure to run preprocess_anndata() first"

        if processed_fcfg:
            assert (
                self.processed_fcfg is not False
            ), "Please make sure to preprocess the cell representation at least once first"

        if labels:
            assert getattr(self, "_labels", None) is not None, "labels not setup yet, run prepare_labels() first"

        if splits:
            assert (
                getattr(self, "_splits", None) is not None
            ), "splits not setup yet, run prepare_dataset_loaders() first"

        return meth(self, *args, **kwargs)

    return bounded


class SpecialTokenMixin:
    _SPECIAL_TOKENS = ["pad", "mask"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.special_tokens = {token: self.adata.n_vars + i for i, token in enumerate(self._SPECIAL_TOKENS)}


class CellRepresentation(SpecialTokenMixin):
    def __init__(self, config, auto_setup: bool = True):
        """Initialize the Cell Rep object with configuration and AnnData object.

        Parameters:
        config (dict): Configuration dictionary.

        """
        self._cfg = config

        self.dataset_preproc_cfg = config.dataset.preprocess_args
        self.dataset_task_cfg = config.tasks.args
        self.fg_cfg = config.fg
        self.fc_cfg = config.fc
        self.fe_cfg = config.fe
        self.model_cfg = config.model
        self.optimizer_cfg = config.optimizer
        self.trainer_cfg = config.trainer
        self.scheduler_cfg = config.scheduler
        self.float_dtype = config.float_dtype
        self.adata = None
        self.processed_fcfg = False

        if auto_setup:
            self.preprocess_anndata()
            self.tokenize_cells()
            self.prepare_dataset_loaders()

        super().__init__()

    # @property
    # @check_states(adata=True, processed_fcfg=True)
    # def cell_representations(self) -> NDArray[np.float32]:
    #     return self.fc[:]

    @property
    @check_states(labels=True)
    def labels(self) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        return self._labels

    @property
    @check_states(labels=True)
    def num_tasks(self) -> int:
        if "_num_tasks" not in self.__dict__:
            warnings.warn(
                "Need to improve to explicitly handle multiclass vs. multilabel",
                UserWarning,
                stacklevel=2,
            )
            assert self.dataset_task_cfg.task_type in [
                "regression",
                "binary",
                "multiclass",
            ], "task type must be regression, binary, or multiclass. Check the task config file."

            task_type = self.dataset_task_cfg.task_type
            if task_type == "regression":
                if len(self.labels.shape) == 1:
                    out = 1
                else:
                    out = self._labels.shape[1]
            elif task_type == "binary":
                if len(self.labels.shape) == 1:
                    out = 1
                else:
                    out = self._labels.shape[1]
            elif task_type == "multiclass":
                out = self._labels.max() + 1
            else:
                raise ValueError(
                    f"Unknown task type {task_type!r}. Valid options are: 'multiclass', 'binary', 'regression'.",
                )

            self._num_tasks = out = int(out)
            print(
                f"> Task dimension: {out} " f"(task type {self.dataset_task_cfg.task_type!r}, {self.labels.shape=})",
            )

        return self._num_tasks

    @property
    @check_states(splits=True)
    def splits(self) -> Dict[str, NDArray[np.int_]]:
        return self._splits

    def convert_to_ensembl_ids(self, data_dir, species="human"):
        """Converts gene symbols in the anndata object to Ensembl IDs using a
        provided mapping.

        Args:
            - data: anndata object with gene symbols as var index
            - data_dir: directory where the data is stored
            - species: species name (default is "human")

        Returns:
            - data: anndata object with Ensembl IDs as var index
            - symbol_to_ensembl_mapping: mapping dictionary from symbols to Ensembl IDs

        """
        symbol_to_ensembl_mapping = symbol_to_ensembl_from_ensembl(
            data_dir=data_dir,
            genes=self.adata.var.index.tolist(),
            species=species,
        )

        self.adata.uns["gene_mapping:symbol_to_ensembl"] = symbol_to_ensembl_mapping.mapping_full

        self.adata.var["gene_symbol"] = self.adata.var.index
        self.adata.var["gene_ensembl"] = self.adata.var["gene_symbol"].map(
            symbol_to_ensembl_mapping.mapping_combined.get,
        )
        self.adata.var.index = self.adata.var.index.map(symbol_to_ensembl_mapping.mapping_reduced)
        self.adata.var.index.name = "index"

        return self.adata, symbol_to_ensembl_mapping

    def get_preprocessed_data_path(self):
        preprocessed_data_path = preprocessed_cfg_path = cfg = None
        if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
            cfg = DictConfig(OmegaConf.to_container(self._cfg, resolve=True))
            preprocessed_data_path, preprocessed_cfg_path = get_cached_paths(
                cfg,
                Path(cache_dir).resolve() / self._cfg.dataset.dataset_name / "preprocessed_anndata",
                "data.h5ad",
            )

        return preprocessed_data_path, preprocessed_cfg_path, cfg

    def anndata_from_cache(self, preprocessed_data_path, preprocessed_cfg_path, cfg):
        if preprocessed_data_path.is_file():
            loaded_cfg_str = OmegaConf.to_yaml(OmegaConf.load(preprocessed_cfg_path)).replace("\n", "\n    ")
            print(f"> Found already preprocessed anndata: {preprocessed_data_path}")
            print(f"  Preprocessing config:\n    {loaded_cfg_str}")
            self.adata = ad.read_h5ad(preprocessed_data_path)
            print(f"> Finished Processing Anndata Object:\n{self.adata}")
            return True

        OmegaConf.save(cfg, preprocessed_cfg_path)

        return False

    def anndata_to_cache(self, preprocessed_data_path):
        print("> Writing preprocessed Anndata Object")
        self.adata.write(preprocessed_data_path)
        print("> Finished writing preprocessed Anndata Object")

    def preprocess_anndata(self):
        if self.adata is not None:
            raise ValueError("Anndata object already exists, are you sure you want to reprocess again?")

        preprocessed_data_path, preprocessed_cfg_path, cfg = self.get_preprocessed_data_path()
        if preprocessed_data_path is not None:
            is_cached = self.anndata_from_cache(preprocessed_data_path, preprocessed_cfg_path, cfg)
            if is_cached:
                return

        self.adata = ad.read_h5ad(self.dataset_preproc_cfg.data_path)
        print(f"> Finished Loading in {self.dataset_preproc_cfg.data_path}")

        # convert gene names to ensembl ids
        if (self.adata.var.index.str.startswith("ENS").sum() / self.adata.n_vars) < 0.9:
            self.adata, symbol_to_ensembl_mapping = self.convert_to_ensembl_ids(
                data_dir=self._cfg.ensembl_dir,
                species=self.dataset_preproc_cfg.species,
            )

        if get_value(self.dataset_preproc_cfg, "normalize"):
            # Normalizing based on target sum
            print("> Normalizing anndata...")
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            assert (
                self.dataset_preproc_cfg.normalize and self.dataset_preproc_cfg.log_1p
            ), "Normalize and Log1P both need to be TRUE"
        else:
            print("> Skipping Normalizing anndata...")

        if get_value(self.dataset_preproc_cfg, "log_1p"):
            # log Transform step
            print("> Log Transforming anndata...")
            sc.pp.log1p(self.adata)
        else:
            print("> Skipping Log Transforming anndata..")

        if get_value(self.dataset_preproc_cfg, "top_n_genes") and self.dataset_preproc_cfg["top_n_genes"] != "false":
            # Identify highly variable genes
            print(f"> Using highly variable subset... top {self.dataset_preproc_cfg.top_n_genes} genes")
            sc.pp.highly_variable_genes(self.adata, n_top_genes=self.dataset_preproc_cfg.top_n_genes)
            self.adata = self.adata[:, self.adata.var["highly_variable"]]
        else:
            print("> No highly variable subset... using entire dataset")

        if get_value(self.dataset_preproc_cfg, "scale_data"):
            # Scale the data
            raise NotImplementedError("Scaling the data is NOT RECOMMENDED, please set it to false")
            print("> Scaling the data...")
            sc.pp.scale(self.adata, max_value=10)
        else:
            print("> Not Scaling the data...")

        if get_value(self.dataset_preproc_cfg, "get_medians"):
            # Get medians
            print("> Getting nonzero medians...")
            csc_expression = csc_array(self.adata.X)
            genewise_nonzero_expression = np.split(csc_expression.data, csc_expression.indptr[1:-1])
            gene_medians = np.array([np.median(gene_nonzeros) for gene_nonzeros in genewise_nonzero_expression])
            self.adata.var["medians"] = gene_medians

        print("> Finished Processing Anndata Object")

        if preprocessed_data_path is not None:
            self.anndata_to_cache(preprocessed_data_path)

        print(f"> Finished Processing Anndata Object:\n{self.adata}")

    @check_states(adata=True, processed_fcfg=True)
    def prepare_dataset_loaders(self):
        # Set up full dataset given the processed cell representation data
        # This will prepare: labels, splits
        full_dataset: Dataset = instantiate_from_config(self._cfg.tasks.args.dataset_config, self)
        self.datasets = {"full": full_dataset}

        # Set up dataset splits given the data splits
        for split, split_idx in self.splits.items():
            self.datasets[split] = Subset(full_dataset, split_idx)

        # Set up data loaders
        dataloader_kwargs = {}  # TODO: USE THIS IF DEBUGGING
        # dataloader_kwargs = {"num_workers": 4}  # TODO: we can parse additional data loader kwargs from config
        self.dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=self.dataset_task_cfg.batchsize,
                shuffle=self.dataset_task_cfg.shuffle if split == "train" else False,
                collate_fn=heimdall_collate_fn,
                **dataloader_kwargs,
            )
            for split, dataset in self.datasets.items()
        }

        dataset_str = pformat(self.datasets).replace("\n", "\n\t")
        print(f"> Finished setting up datasets (and loaders):\n\t{dataset_str}")

    def rebalance_dataset(self, df):
        # Step 1: Find which label has a lower number
        label_counts = df["labels"].value_counts()
        minority_label = label_counts.idxmin()
        majority_label = label_counts.idxmax()
        minority_count = label_counts[minority_label]

        print(f"Minority label: {minority_label}")
        print(f"Majority label: {majority_label}")
        print(f"Number of samples in minority class: {minority_count}")

        # Step 2: Downsample the majority class
        df_minority = df[df["labels"] == minority_label]
        df_majority = df[df["labels"] == majority_label]

        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=minority_count,
            random_state=42,
        )

        # Combine minority class with downsampled majority class
        df_balanced = pd.concat([df_minority, df_majority_downsampled])

        print(f"Original dataset shape: {df.shape}")
        print(f"Balanced dataset shape: {df_balanced.shape}")
        print("New label distribution:")
        print(df_balanced["labels"].value_counts())

        return df_balanced

    def drop_invalid_genes(self):
        """Modify `self.adata` to only contain valid genes after preprocessing
        with the `Fg`."""

        valid_mask = self.adata.var["identity_valid_mask"]
        self.adata.raw = self.adata.copy()
        self.adata = self.adata[:, valid_mask].copy()

        # setting up some pointers so that they all reference the same anndata object
        self.fe.adata = self.adata
        self.fc.adata = self.adata

        preprocessed_data_path, *_ = self.get_preprocessed_data_path()
        if preprocessed_data_path is not None:
            self.anndata_to_cache(preprocessed_data_path)

    def load_tokenization_from_cache(self, cache_dir):
        cfg = DictConfig(
            {key: OmegaConf.to_container(getattr(self, key), resolve=True) for key in ("fg_cfg", "fe_cfg", "fc_cfg")},
        )
        processed_data_path, processed_cfg_path = get_cached_paths(
            cfg,
            Path(cache_dir).resolve() / self._cfg.dataset.dataset_name / "processed_data",
            "data.pkl",
        )
        if processed_data_path.is_file():
            loaded_cfg_str = OmegaConf.to_yaml(OmegaConf.load(processed_cfg_path)).replace("\n", "\n    ")
            print(f"> Using processed cell representations: {processed_data_path}")
            print(f"  Processing config:\n    {loaded_cfg_str}")

            with open(processed_data_path, "rb") as rep_file:
                (
                    identity_embedding_index,
                    identity_valid_mask,
                    gene_embeddings,
                    expression_embeddings,
                ) = pkl.load(rep_file)

                self.fg.load_from_cache(identity_embedding_index, identity_valid_mask, gene_embeddings)
                self.fe.load_from_cache(expression_embeddings)

                self.processed_fcfg = True

            return True

        OmegaConf.save(cfg, processed_cfg_path)
        return False

    def save_tokenization_to_cache(self, cache_dir):
        cfg = DictConfig(
            {key: OmegaConf.to_container(getattr(self, key), resolve=True) for key in ("fg_cfg", "fe_cfg", "fc_cfg")},
        )
        processed_data_path, processed_cfg_path = get_cached_paths(
            cfg,
            Path(cache_dir).resolve() / self._cfg.dataset.dataset_name / "processed_data",
            "data.pkl",
        )

        # Gather things for caching
        identity_embedding_index, identity_valid_mask = self.fg.__getitem__(self.adata.var_names, return_mask=True)

        gene_embeddings = self.fg.gene_embeddings
        expression_embeddings = self.fe.expression_embeddings

        with open(processed_data_path, "wb") as rep_file:
            cache_representation = (
                identity_embedding_index,
                identity_valid_mask,
                gene_embeddings,
                expression_embeddings,
            )
            pkl.dump(cache_representation, rep_file)
            print(f"Finished writing cell representations at {processed_data_path}")

    def instantiate_representation_functions(self):
        """Instantiate `f_g`, `fe` and `f_c` according to config."""
        self.fg: Fg
        self.fe: Fe
        self.fc: Fc
        self.fg, fg_name = instantiate_from_config(
            self.fg_cfg,
            self.adata,
            vocab_size=self.adata.n_vars + 2,
            return_name=True,
        )
        self.fe, fe_name = instantiate_from_config(
            self.fe_cfg,
            self.adata,
            vocab_size=self.adata.n_vars + 2,
            return_name=True,
        )
        self.fc, fc_name = instantiate_from_config(
            self.fc_cfg,
            self.fg,
            self.fe,
            self.adata,
            float_dtype=self.float_dtype,
            return_name=True,
        )

    @check_states(adata=True)
    def tokenize_cells(self):
        """Processes the `f_g`, `fe` and `f_c` from the config.

        This will first check to see if the cell representations are already
        cached, and then will either load the cached representations or compute
        them and save them.

        """

        self.instantiate_representation_functions()

        if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
            is_cached = self.load_tokenization_from_cache(cache_dir)
            if is_cached:
                return

        self.fg.preprocess_embeddings()
        print(f"> Finished calculating fg with {self.fg_cfg.type}")

        self.drop_invalid_genes()  # TODO: remove this if necessary
        print("> Finished dropping invalid genes from AnnData")

        self.fe.preprocess_embeddings()
        print(f"> Finished calculating fe with {self.fe_cfg.type}")

        self.processed_fcfg = True

        if (self._cfg.cache_preprocessed_dataset_dir) is not None:
            self.save_tokenization_to_cache(cache_dir)
