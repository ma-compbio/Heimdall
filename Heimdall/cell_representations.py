"""The Cell Representation Object for Processing."""

import os
import warnings
from abc import ABC, abstractmethod
from functools import partial, wraps
from pprint import pformat
from typing import Callable, Dict, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from torch.utils.data import Subset
from tqdm import tqdm

from Heimdall.utils import deprecate, get_value, heimdall_collate_fn
from notebooks import data_processing_utils

SPLIT_MASK_KEYS = {"full_mask", "train_mask", "val_mask", "test_mask", "full", "train", "val", "test"}

CellFeatType = Union[NDArray[np.int_], NDArray[np.float32]]
FeatType = Union[CellFeatType, Tuple[CellFeatType, CellFeatType]]
LabelType = Union[NDArray[np.int_], NDArray[np.float32]]


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
            ), "splits not setup yet, run prepare_datase_loaders() first"

        return meth(self, *args, **kwargs)

    return bounded


class Dataset(PyTorchDataset, ABC):
    def __init__(self, data: "CellRepresentation"):
        super().__init__()
        self._data = data

        if self.labels is None:
            self._setup_labels()

        # NOTE: need to setup labels first, index sizes might depend on it
        self._setup_idx()

        if self.splits is None:
            self._setup_splits()

    @property
    def idx(self) -> NDArray[np.int_]:
        return self._idx

    @property
    def data(self) -> "CellRepresentation":
        return self._data

    @property
    def labels(self) -> NDArray:
        return getattr(self.data, "_labels", None)

    @labels.setter
    def labels(self, val):
        self.data._labels = val

    @property
    def splits(self) -> NDArray:
        return getattr(self.data, "_splits", None)

    @splits.setter
    def splits(self, val):
        self.data._splits = val

    def __len__(self) -> int:
        return len(self.idx)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(size={len(self):,}) wrapping: {self.data}"

    def _setup_splits(self):
        # TODO: use predefined splits if available
        predefined_splits = None
        size = len(self)
        seed = self.data._cfg.seed

        if predefined_splits is None:
            warnings.warn(
                "Pre-defined split unavailable, using random 6/2/2 split",
                UserWarning,
                stacklevel=2,
            )
            train_val_idx, test_idx = train_test_split(np.arange(size), train_size=0.6, random_state=seed)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=seed)

        self.splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    @abstractmethod
    def _setup_idx(self): ...

    @abstractmethod
    def _setup_labels(self): ...

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[FeatType, LabelType]: ...


class SingleInstanceDataset(Dataset):
    def _setup_idx(self):
        self._idx = np.arange(self.data.adata.shape[0])

    def _setup_labels(self):
        adata = self.data.adata
        dataset_task_cfg = self.data.dataset_task_cfg

        df = adata.obs
        class_mapping = {
            label: idx
            for idx, label in enumerate(
                df[dataset_task_cfg.label_col_name].unique(),
                start=0,
            )
        }
        df["class_id"] = df[dataset_task_cfg.label_col_name].map(class_mapping)
        self.labels = np.array(df["class_id"])

    def __getitem__(self, idx) -> Tuple[CellFeatType, LabelType]:
        return {
            "inputs": self.data.cell_representations[idx],
            "labels": self.data.labels[idx],
        }


class PairedInstanceDataset(Dataset):
    def _setup_idx(self):
        # NOTE: full mask is set up during runtime given split masks or the data
        mask = self.data.adata.obsp["full_mask"]
        self._idx = np.vstack(np.nonzero(mask)).T  # pairs x 2

    def _setup_labels(self):
        adata = self.data.adata
        dataset_task_cfg = self.data.dataset_task_cfg

        all_obsp_task_keys, obsp_mask_keys = [], []
        for key in adata.obsp:
            (obsp_mask_keys if key in SPLIT_MASK_KEYS else all_obsp_task_keys).append(key)
        all_obsp_task_keys = sorted(all_obsp_task_keys)

        # Select task keys
        candidate_obsp_task_keys = dataset_task_cfg.interaction_type
        if candidate_obsp_task_keys == "_all_":
            obsp_task_keys = all_obsp_task_keys
        else:
            if isinstance(candidate_obsp_task_keys, str):
                candidate_obsp_task_keys = [candidate_obsp_task_keys]

            if invalid_obsp_task_keys := [i for i in candidate_obsp_task_keys if i not in all_obsp_task_keys]:
                raise ValueError(
                    f"{len(invalid_obsp_task_keys)} out of {len(candidate_obsp_task_keys)} "
                    f"specified interaction types are invalid: {invalid_obsp_task_keys}\n"
                    f"Valid options are: {pformat(all_obsp_task_keys)}",
                )

            obsp_task_keys = candidate_obsp_task_keys

        # Set up task mask
        full_mask = np.sum([np.abs(adata.obsp[i]) for i in obsp_task_keys], axis=-1) > 0
        adata.obsp["full_mask"] = full_mask
        nz = np.nonzero(full_mask)
        num_tasks = len(obsp_task_keys)

        # TODO: specify task type multiclass/multilabel/regression in config
        (labels := np.empty((len(nz[0]), num_tasks), dtype=np.float32)).fill(np.nan)
        for i, task in enumerate(obsp_task_keys):
            label_i = np.array(adata.obsp[task][nz]).ravel()
            labels[:, i][label_i == 1] = 1
            labels[:, i][label_i == -1] = 0
        self.labels = labels

    def __getitem__(self, idx) -> Tuple[Tuple[CellFeatType, CellFeatType], LabelType]:
        cell1_idx, cell2_idx = self.idx[idx]
        return {
            "inputs": (self.data.cell_representations[cell1_idx], self.data.cell_representations[cell2_idx]),
            "labels": self.data.labels[idx],
        }


class CellRepresentation:
    def __init__(self, config, auto_setup: bool = True):
        """Initialize the Cell Rep object with configuration and AnnData object.

        Parameters:
        config (dict): Configuration dictionary.

        """
        self._cfg = config

        self.dataset_preproc_cfg = config.dataset.preprocess_args
        self.dataset_task_cfg = config.tasks.args
        self.fg_cfg = config.f_g
        self.fc_cfg = config.f_c
        self.model_cfg = config.model
        self.optimizer_cfg = config.optimizer
        self.trainer_cfg = config.trainer
        self.scheduler_cfg = config.scheduler
        self.adata = None
        self.task_structure = config.tasks.args.task_structure
        self.processed_fcfg = False

        if auto_setup:
            # TODO: get from config
            from Heimdall.f_c import old_geneformer_fc
            from Heimdall.f_g import identity_fg

            self.preprocess_anndata()
            self.preprocess_f_g(identity_fg)
            self.preprocess_f_c(old_geneformer_fc)
            self.prepare_dataset_loaders()

    @property
    @check_states(adata=True, processed_fcfg=True)
    def cell_representations(self) -> NDArray[np.float32]:
        return self.adata.layers["cell_representation"]

    @property
    @check_states(labels=True)
    def labels(self) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        return self._labels

    @property
    @check_states(labels=True)
    def num_tasks(self) -> int:
        warnings.warn(
            "Need to improve to explicitly handle multiclass vs. multilabel",
            UserWarning,
            stacklevel=2,
        )
        if (self.labels % 1).any():  # inferred to be regression
            task_type = "regression"
            out = self._labels.shape[1]
        elif self.labels.max() == 1:  # inferred to be multilabel
            task_type = "classification-multilabel"
            if len(self.labels.shape) == 1:
                out = 1
            else:
                out = self._labels.shape[1]
        else:  # inferred to be multiclass
            task_type = "classification-multiclass"
            out = self._labels.max() + 1

        print(f"> Task dimension inferred: {out} (inferred task type {task_type!r}, {self.labels.shape=})")

        return out

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
        if species == "mouse":
            self.adata.var_names = self.adata.var_names.str.upper()
        symbol_to_ensembl_mapping = data_processing_utils.symbol_to_ensembl_from_ensembl(
            data_dir=data_dir,
            genes=self.adata.var.index.tolist(),
            species="human",
        )

        self.adata.uns["gene_mapping:symbol_to_ensembl"] = symbol_to_ensembl_mapping.mapping_full

        self.adata.var["gene_symbol"] = self.adata.var.index
        self.adata.var["gene_ensembl"] = self.adata.var["gene_symbol"].map(
            symbol_to_ensembl_mapping.mapping_combined.get,
        )
        self.adata.var.index = self.adata.var.index.map(symbol_to_ensembl_mapping.mapping_reduced)
        self.adata.var.index.name = "index"

        return self.adata, symbol_to_ensembl_mapping

    def preprocess_anndata(self, cache_and_load_preproc=False):
        if self.adata is not None:
            raise ValueError("Anndata object already exists, are you sure you want to reprocess again?")

        os.makedirs("heimdall_preprocessed", exist_ok=True)

        if cache_and_load_preproc:
            filepath = self.dataset_preproc_cfg.data_path.split("/")
            preprocessed_data_path = "heimdall_preprocessed/preprocessed_" + filepath[-1]

            if os.path.isfile(preprocessed_data_path):
                self.adata = ad.read_h5ad(preprocessed_data_path)
                print(f"> Found already preprocessed dataset, loading in {preprocessed_data_path}")
                print(f"> Finished Loading in preprocessed dataset: {preprocessed_data_path}")
                self.sequence_length = len(self.adata.var)
                return

            else:
                self.adata = ad.read_h5ad(self.dataset_preproc_cfg.data_path)
                print(f"> Finished Loading in {self.dataset_preproc_cfg.data_path}")
        else:
            self.adata = ad.read_h5ad(self.dataset_preproc_cfg.data_path)
            print(f"> Finished Loading in {self.dataset_preproc_cfg.data_path}")

        # convert gene names to ensembl ids
        if (self.adata.var.index.str.startswith("ENS").sum() / len(self.adata.var.index)) < 0.9:
            self.adata, symbol_to_ensembl_mapping = self.convert_to_ensembl_ids(
                data_dir=self._cfg.ensembl_dir,
                species=self.dataset_preproc_cfg.species,
            )

        # check if layer already exists with your desired preprocessing
        preprocessing_string = "_".join(
            [g for g in self.dataset_preproc_cfg.keys() if get_value(self.dataset_preproc_cfg, g)],
        )
        if preprocessing_string in self.adata.layers.keys():
            print("> Using cached preprocessed data")
            self.adata.X = self.adata.layers[preprocessing_string].copy()
            self.sequence_length = len(self.adata.var)
            return

        else:
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

            if (
                get_value(self.dataset_preproc_cfg, "top_n_genes")
                and self.dataset_preproc_cfg["top_n_genes"] != "false"
            ):
                print(self.dataset_preproc_cfg)
                # Identify highly variable genes
                print(f"> Using highly variable subset... top {self.dataset_preproc_cfg.top_n_genes} genes")
                sc.pp.highly_variable_genes(self.adata, n_top_genes=self.dataset_preproc_cfg.top_n_genes)
                self.adata = self.adata[:, self.adata.var["highly_variable"]]
            else:
                print("> No highly variable subset... using entire dataset")

            self.sequence_length = len(self.adata.var)

            if get_value(self.dataset_preproc_cfg, "scale_data"):
                # Scale the data
                print("> Scaling the data...")
                sc.pp.scale(self.adata, max_value=10)
            else:
                print("> Not Scaling the data...")

            print("> Finished Processing Anndata Object")
            print("> Writing preprocessed Anndata Object")
            self.adata.layers[preprocessing_string] = self.adata.X.copy()

            if cache_and_load_preproc:
                self.adata.write(preprocessed_data_path)
                print("> Finished writing preprocessed Anndata Object")

            print("> Not Scaling the data...")

        print(f"> Finished Processing Anndata Object:\n{self.adata}")

    @check_states(adata=True, processed_fcfg=True)
    def prepare_dataset_loaders(self):
        # Set up full dataset given the processed cell representation data
        # This will prepare: labels, splits
        if self.task_structure == "single":
            full_dataset = SingleInstanceDataset(self)
        elif self.task_structure == "paired":
            full_dataset = PairedInstanceDataset(self)
        else:
            raise ValueError("config.tasks.args.task_structure must be 'single' or 'paired'")
        self.datasets = {"full": full_dataset}

        # Set up dataset splits given the data splits
        for split, split_idx in self.splits.items():
            self.datasets[split] = Subset(full_dataset, split_idx)

        # Set up data loaders
        dataloader_kwargs = {}  # TODO: we can parse additional data loader kwargs from config
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

    @deprecate(raise_error=True)
    def _prepare_splits(self):
        # TODO: use predefined splits if available
        predefined_splits = None
        size = len(self.datasets["full"])
        seed = self._cfg.seed

        if predefined_splits is None:
            warnings.warn(
                "Pre-defined split unavailable, using random 6/2/2 split",
                UserWarning,
                stacklevel=2,
            )
            train_val_idx, test_idx = train_test_split(np.arange(size), train_size=0.6, random_state=seed)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=seed)

        self._splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    @deprecate
    def prepare_datasets(self):
        """After preprocessing, provides the dataset in dataframe format that
        can be processed."""
        assert self.adata is not None, "no adata found, Make sure to run preprocess_anndata() first"
        assert (
            self.processed_fcfg is not False
        ), "Please make sure to preprocess the cell representation at least once first"

        cell_representation = self.adata.layers["cell_representation"]

        if self.task_structure == "single":
            self.prepare_labels()
            x = cell_representation
            y = self.labels
            self.df = pd.DataFrame({"inputs": x, "labels": y})

        elif self.task_structure == "paired":
            self.df = self.prepare_paired_dataset(self.dataset_task_cfg.interaction_type)

            if self.dataset_task_cfg.rebalance:
                self.df = self.rebalance_dataset(self.df)

        else:
            raise ValueError("config.tasks.args.task_structure must be 'single' or 'paired'")

        print("> Finished Preprocessing the dataset into self.df ")

    @deprecate(raise_error=True)
    def prepare_labels(self):
        """Pull out the specified class from data to set up label."""
        assert self.adata is not None, "no adata found, Make sure to run preprocess_anndata() first"

        if self.task_structure == "single":
            df = self.adata.obs
            class_mapping = {
                label: idx
                for idx, label in enumerate(
                    df[self.dataset_task_cfg.label_col_name].unique(),
                    start=0,
                )
            }
            df["class_id"] = df[self.dataset_task_cfg.label_col_name].map(class_mapping)
            self._labels = np.array(df["class_id"])

        elif self.task_structure == "paired":
            all_obsp_task_keys, obsp_mask_keys = [], []
            for key in self.adata.obsp:
                (obsp_mask_keys if key in SPLIT_MASK_KEYS else all_obsp_task_keys).append(key)
            all_obsp_task_keys = sorted(all_obsp_task_keys)

            # Select task keys
            candidate_obsp_task_keys = self.dataset_task_cfg.interaction_type
            if candidate_obsp_task_keys == "_all_":
                obsp_task_keys = all_obsp_task_keys
            else:
                if isinstance(candidate_obsp_task_keys, str):
                    candidate_obsp_task_keys = [candidate_obsp_task_keys]

                if invalid_obsp_task_keys := [i for i in candidate_obsp_task_keys if i not in all_obsp_task_keys]:
                    raise ValueError(
                        f"{len(invalid_obsp_task_keys)} out of {len(candidate_obsp_task_keys)} "
                        f"specified interaction types are invalid: {invalid_obsp_task_keys}\n"
                        f"Valid options are: {pformat(all_obsp_task_keys)}",
                    )

            # Set up task mask
            full_mask = np.sum([np.abs(self.adata.obsp[i]) for i in obsp_task_keys], axis=-1) > 0
            self.adata.obsp["full_mask"] = full_mask
            nz = np.nonzero(full_mask)
            num_tasks = len(obsp_task_keys)

            # TODO: specify task type multiclass/multilabel/regression in config
            (labels := np.empty((len(nz[0]), num_tasks), dtype=np.float32)).fill(np.nan)
            for i, task in enumerate(obsp_task_keys):
                label_i = np.array(self.adata.obsp[task][nz]).ravel()
                labels[:, i][label_i == 1] = 1
                labels[:, i][label_i == -1] = 0
            self._labels = labels

            # TODO: set up split given the predefiend split mask if possible

        else:
            raise ValueError("config.tasks.args.task_structure must be 'single' or 'paired'")

        print(f"> Finished extracting labels, self.labels.shape: {self.labels.shape}")

    @deprecate(raise_error=True)
    def prepare_paired_dataset(self, interaction_type):
        assert self.adata is not None, "no adata found, Make sure to run preprocess_anndata() first"

        interaction_matrix = self.adata.obsp[interaction_type]
        cell_expression = self.adata.layers["cell_representation"]

        # Ensure interaction_matrix is in CSR format for efficient row slicing
        if not isinstance(interaction_matrix, csr_matrix):
            interaction_matrix = csr_matrix(interaction_matrix)

        # Initialize lists to store data
        cell_a_indices = []
        cell_b_indices = []
        labels = []

        # Iterate through non-zero elements efficiently
        for i in tqdm(range(interaction_matrix.shape[0]), desc=f"Processing {interaction_type}", unit="cell"):
            row = interaction_matrix.getrow(i)
            non_zero_cols = row.nonzero()[1]
            non_zero_values = row.data

            cell_a_indices.extend([i] * len(non_zero_cols))
            cell_b_indices.extend(non_zero_cols)
            labels.extend(non_zero_values)

        # Create DataFrame with indices
        df = pd.DataFrame(
            {
                "CellA_Index": cell_a_indices,
                "CellB_Index": cell_b_indices,
                "labels": labels,
            },
        )
        # Add expression data
        if issparse(cell_expression):
            df["CellA_Expression"] = [cell_expression[i].toarray().flatten() for i in df["CellA_Index"]]
            df["CellB_Expression"] = [cell_expression[j].toarray().flatten() for j in df["CellB_Index"]]
        else:
            df["CellA_Expression"] = [cell_expression[i] for i in df["CellA_Index"]]
            df["CellB_Expression"] = [cell_expression[j] for j in df["CellB_Index"]]

        df["labels"] = df["labels"].replace(-1, 0)

        return df[["CellA_Expression", "CellB_Expression", "labels"]]

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

    def preprocess_f_g(self, f_g):
        """Process f_g.

        Run the f_g, and then preprocess and store it locally the f_g must
        return a `gene_mapping` where the keys are the gene ids and the ids are
        the.

        The f_g will take in the anndata as an object just for flexibility

        For example:
        ```
        {'Sox17': 0,
        'Rgs20': 1,
        'St18': 2,
        'Cpa6': 3,
        'Prex2': 4,
        ...
        }

        or even:
        {'Sox17': [0.3, 0.1. 0.2 ...],
        'Rgs20':  [0.3, 0.1. 0.2 ...],
        'St18':  [0.3, 0.1. 0.2 ...],
        'Cpa6':  [0.3, 0.1. 0.2 ...],
        'Prex2':  [0.3, 0.1. 0.2 ...],
        ...
        }
        ```

        """
        assert self.fg_cfg.args.output_type in ["ids", "vector"]

        # For identity, we convert each of the vars into a unique ID
        gene_mapping = f_g(self.adata.var)

        assert all(
            isinstance(value, (np.ndarray, list, int)) for value in gene_mapping.values()
        ), "Make sure that all values in the gene_mapping dictionary are either int, list or np array"

        self.f_g = gene_mapping
        print(f"> Finished calculating f_g with {self.fg_cfg.name}")
        return

    def preprocess_f_c(self, f_c):
        """Process f_c.

        Preprocess the cell f_c, this will preprocess the anndata.X into the
        actual dataset to the actual tokenizers, you can imagine this as a cell
        tokenizer.

        The f_c will take as input the f_g, then the anndata, then the

        """
        self.adata = f_c(self.f_g, self.adata)
        print(f"> Finished calculating f_c with {self.fc_cfg.name}")
        self.processed_fcfg = True
        return
