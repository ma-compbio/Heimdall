"""The Cell Representation Object for Processing."""

import pickle as pkl
from collections import defaultdict
from functools import partial, wraps
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from accelerate import Accelerator
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from scipy.sparse import csc_array
from sklearn.utils import resample
from torch.utils.data import DataLoader, Subset

from Heimdall.datasets import Dataset, PartitionedSubset
from Heimdall.fc import Fc
from Heimdall.fe import Fe
from Heimdall.fg import Fg
from Heimdall.samplers import PartitionedBatchSampler, PartitionedDistributedSampler
from Heimdall.task import Tasklist

# from Heimdall.samplers import PartitionedDistributedSampler
from Heimdall.utils import (
    convert_to_ensembl_ids,
    get_cached_paths,
    get_value,
    heimdall_collate_fn,
    instantiate_from_config,
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
            assert getattr(self, "_labels", None) is not None, "labels not setup yet, create `Dataset` object first"

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
    def __init__(self, config, accelerator: Accelerator, auto_setup: bool = True):
        """Initialize the Cell Rep object with configuration and AnnData object.

        Parameters:
        config (dict): Configuration dictionary.

        """
        self.rank = accelerator.process_index
        self.num_replicas = accelerator.num_processes
        self.accelerator = accelerator

        self.cr_setup = False
        self._cfg = config

        self.dataset_preproc_cfg = config.dataset.preprocess_args
        if hasattr(config.tasks.args, "subtask_configs"):
            self.tasklist = instantiate_from_config(config.tasks, self)
        else:
            self.tasklist = Tasklist(self, subtask_configs={"default": config.tasks})
            # task = instantiate_from_config(config.tasks, self)
            # self.tasklist["default"] = task

        self.num_subtasks = self.tasklist.num_subtasks

        self.fg_cfg = config.fg
        self.fc_cfg = config.fc
        self.fe_cfg = config.fe
        self.float_dtype = config.float_dtype
        self.adata = None
        self.processed_fcfg = False

        seed = 0  # TODO: make this configurable???
        self.rng = np.random.default_rng(seed)

        if auto_setup:
            self.setup()
            self.prepare_full_dataset()
            self.prepare_dataset_loaders()

    def setup(self):
        self.preprocess_anndata()
        self.tokenize_cells()
        super().__init__()
        # if hasattr(self, "datasets") and "full" in self.datasets:
        #     self.prepare_dataset_loaders()
        self.cr_setup = True

    @property
    @check_states(labels=True)
    def labels(self) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        labels = {subtask_name: subtask.labels for subtask_name, subtask in self.tasklist}
        return labels

    @labels.setter
    def labels(self, val) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        for subtask_name, subtask in self.tasklist:
            subtask.labels = val[subtask_name]

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

        _, gene_mapping = convert_to_ensembl_ids(self.adata, data_dir, species=species)
        return self.adata, gene_mapping

    def get_preprocessed_data_path(self):
        preprocessed_data_path = preprocessed_cfg_path = cfg = None
        if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:

            cfg = DictConfig({"dataset": OmegaConf.to_container(self._cfg.dataset, resolve=True)})

            preprocessed_data_path, preprocessed_cfg_path = get_cached_paths(
                cfg,
                Path(cache_dir).resolve() / self._cfg.dataset.dataset_name / "preprocessed_anndata",
                "data.h5ad",
            )

        return preprocessed_data_path, preprocessed_cfg_path, cfg

    def anndata_from_cache(self, preprocessed_data_path, preprocessed_cfg_path, cfg):
        if preprocessed_data_path.is_file():
            self.check_print(
                f"> Found already preprocessed anndata: {preprocessed_data_path}",
                cr_setup=True,
                rank=True,
            )
            # loaded_cfg_str = OmegaConf.to_yaml(OmegaConf.load(preprocessed_cfg_path)).replace("\n", "\n    ")
            # print(f"  Preprocessing config:\n    {loaded_cfg_str}") # TODO: add verbosity level
            self.adata = ad.read_h5ad(
                preprocessed_data_path,
                backed="r",
            )  # add backed argument to prevent entire dataset from being read into mem
            self.sequence_length = len(self.adata.var)
            self.check_print(f"> Finished Processing Anndata Object:\n{self.adata}", cr_setup=True, rank=True)
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
        self.check_print(f"> Finished Loading in {self.dataset_preproc_cfg.data_path}", cr_setup=True)
        # convert gene names to ensembl ids
        self.check_print("> Converting gene names to Ensembl IDs...", cr_setup=True)
        self.adata, _ = self.convert_to_ensembl_ids(
            data_dir=self._cfg.ensembl_dir,
            species=self.dataset_preproc_cfg.species,
        )

        if get_value(self.dataset_preproc_cfg, "normalize"):
            self.check_print("> Normalizing AnnData...", cr_setup=True)

            if sparse.issparse(self.adata.X):
                data = self.adata.X.data
            else:
                data = self.adata.X

            # Store mask of NaNs
            nan_mask = np.isnan(data)

            if np.any(nan_mask):
                data[nan_mask] = 0
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            if np.any(nan_mask):
                data[nan_mask] = np.nan  # NOTE: must not be integer-valued

            assert (
                self.dataset_preproc_cfg.normalize and self.dataset_preproc_cfg.log_1p
            ), "Normalize and Log1P both need to be TRUE"
        else:
            self.check_print("> Skipping Normalizing anndata...", cr_setup=True)

        if get_value(self.dataset_preproc_cfg, "log_1p"):
            self.check_print("> Log Transforming anndata...", cr_setup=True)

            sc.pp.log1p(self.adata)
        else:
            self.check_print("> Skipping Log Transforming anndata..", cr_setup=True)

        if get_value(self.dataset_preproc_cfg, "top_n_genes") and self.dataset_preproc_cfg["top_n_genes"] != "false":
            # Identify highly variable genes
            print(f"> Using highly variable subset... top {self.dataset_preproc_cfg.top_n_genes} genes")
            sc.pp.highly_variable_genes(self.adata, n_top_genes=self.dataset_preproc_cfg.top_n_genes)
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()
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

        if preprocessed_data_path is not None:
            self.anndata_to_cache(preprocessed_data_path)

        self.check_print(f"> Finished Processing Anndata Object:\n{self.adata}", cr_setup=True)

    @check_states(adata=True, processed_fcfg=True)
    def prepare_full_dataset(self):
        # Set up full dataset given the processed cell representation data
        # This will prepare: labels, splits
        full_dataset: Dataset = instantiate_from_config(self.tasklist.dataset_config, self)
        self.datasets = {"full": full_dataset}

    @check_states(adata=True, processed_fcfg=True)
    def prepare_dataset_loaders(self):
        full_dataset = self.datasets["full"]
        # Set up dataset splits given the data splits
        for split, split_idx in self.splits.items():
            self.datasets[split] = Subset(full_dataset, split_idx)

        # Set up data loaders
        # dataloader_kwargs = {}  # TODO: USE THIS IF DEBUGGING
        dataloader_kwargs = {"num_workers": 4}  # TODO: we can parse additional data loader kwargs from config
        self.dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=self._cfg.trainer.per_device_batch_size,
                shuffle=self.tasklist.shuffle if split == "train" else False,
                collate_fn=heimdall_collate_fn,
                **dataloader_kwargs,
            )
            for split, dataset in self.datasets.items()
        }

        dataset_str = pformat(self.datasets).replace("\n", "\n\t")
        self.check_print(f"> Finished setting up datasets (and loaders):\n\t{dataset_str}", rank=True)

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

        self.fc.adata = self.adata

        preprocessed_data_path, *_ = self.get_preprocessed_data_path()
        if preprocessed_data_path is not None:
            self.anndata_to_cache(preprocessed_data_path)

        self.check_print(f"> Finished dropping invalid genes, yielding new AnnData: :\n{self.adata}", cr_setup=True)

    def load_tokenization_from_cache(self, cache_dir, hash_vars):
        cfg = DictConfig(
            {key: OmegaConf.to_container(getattr(self, key), resolve=True) for key in ("fg_cfg", "fe_cfg", "fc_cfg")},
        )
        cfg = {**cfg, "hash_vars": hash_vars}
        processed_data_path, processed_cfg_path = get_cached_paths(
            cfg,
            Path(cache_dir).resolve() / self._cfg.dataset.dataset_name / "processed_data",
            "data.pkl",
        )
        if processed_data_path.is_file():
            # loaded_cfg_str = OmegaConf.to_yaml(OmegaConf.load(processed_cfg_path)).replace("\n", "\n    ")
            # print(f"  Processing config:\n    {loaded_cfg_str}") # TODO: add verbosity levels

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

    def save_tokenization_to_cache(self, cache_dir, hash_vars):
        # Gather things for caching
        identity_embedding_index, identity_valid_mask = self.fg.__getitem__(self.adata.var_names, return_mask=True)

        gene_embeddings = self.fg.gene_embeddings
        expression_embeddings = self.fe.expression_embeddings

        cfg = DictConfig(
            {key: OmegaConf.to_container(getattr(self, key), resolve=True) for key in ("fg_cfg", "fe_cfg", "fc_cfg")},
        )
        cfg = {**cfg, "hash_vars": hash_vars}
        processed_data_path, processed_cfg_path = get_cached_paths(
            cfg,
            Path(cache_dir).resolve() / self._cfg.dataset.dataset_name / "processed_data",
            "data.pkl",
        )
        if not processed_data_path.is_file():
            with open(processed_data_path, "wb") as rep_file:
                cache_representation = (
                    identity_embedding_index,
                    identity_valid_mask,
                    gene_embeddings,
                    expression_embeddings,
                )
                pkl.dump(cache_representation, rep_file)
                self.check_print(f"Finished writing cell representations at {processed_data_path}", cr_setup=True)

    def instantiate_representation_functions(self):
        """Instantiate `f_g`, `fe` and `f_c` according to config."""
        self.fg: Fg
        self.fe: Fe
        self.fc: Fc
        self.fg, fg_name = instantiate_from_config(
            self.fg_cfg,
            self.adata,
            vocab_size=self.adata.n_vars + 2,
            rng=self.rng,
            return_name=True,
        )
        self.fe, fe_name = instantiate_from_config(
            self.fe_cfg,
            self.adata,
            vocab_size=self.adata.n_vars + 2,  # TODO: figure out a way to fix the number of expr tokens
            rng=self.rng,
            return_name=True,
        )
        self.fc, fc_name = instantiate_from_config(
            self.fc_cfg,
            self.fg,
            self.fe,
            self.adata,
            float_dtype=self.float_dtype,
            rng=self.rng,
            return_name=True,
        )

    @check_states(adata=True)
    def tokenize_cells(self, hash_vars=()):
        """Processes the `f_g`, `fe` and `f_c` from the config.

        This will first check to see if the cell representations are already
        cached, and then will either load the cached representations or compute
        them and save them.

        """
        self.instantiate_representation_functions()

        if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
            is_cached = self.load_tokenization_from_cache(cache_dir, hash_vars=hash_vars)
            if is_cached:
                return

        self.fg.preprocess_embeddings()
        self.check_print(f"> Finished calculating fg with {self.fg_cfg.type}", cr_setup=True)

        self.drop_invalid_genes()
        self.check_print("> Finished dropping invalid genes from AnnData", cr_setup=True)

        self.fe.preprocess_embeddings()
        self.check_print(f"> Finished calculating fe with {self.fe_cfg.type}", cr_setup=True)

        self.processed_fcfg = True

        if cache_dir is not None:
            self.save_tokenization_to_cache(cache_dir, hash_vars=hash_vars)

    def check_print(self, message, rank=False, cr_setup=False):

        if (not rank or self.rank == 0) and (not cr_setup or not self.cr_setup):
            print(message)


class PartitionedCellRepresentation(CellRepresentation):
    def __init__(self, config, accelerator: Accelerator, auto_setup: bool = True):
        super().__init__(config, accelerator, auto_setup=False)

        # Expect `data_path` to hold parent directory, not filepath
        self.partition_file_paths = sorted(
            Path(self._cfg.dataset.preprocess_args.data_path).glob("*.h5ad"),
        )
        self.num_partitions = len(self.partition_file_paths)

        self.partition_sizes = {}
        if auto_setup:
            for partition in range(self.num_partitions):
                self.partition = partition
                self.partition_sizes[partition] = self.adata.n_obs

            self.cr_setup = True
            self.prepare_full_dataset()
            self.prepare_dataset_loaders()

            self.partition = 0  # TODO: don't hardcode

            SpecialTokenMixin.__init__(self)

    def setup(self):
        self.preprocess_anndata()
        self.tokenize_cells(hash_vars=(self.partition,))

    def close_partition(self):
        """Close current partition."""
        if self.adata is not None:
            self.adata.file.close()

            del self.adata
            self.adata = None

    @property
    def partition(self):
        return self._partition

    @partition.setter
    def partition(self, partition):
        """Move to a new partition."""
        if getattr(self, "_partition", None) == partition:
            return

        self.close_partition()
        self._partition = partition

        # Preprocess partition AnnData
        self.check_print(f"> Opening partition {partition + 1} of {self.num_partitions}", cr_setup=True, rank=True)
        self.dataset_preproc_cfg.data_path = self.partition_file_paths[partition]
        self.setup()

    @check_states(adata=True, processed_fcfg=True)
    def prepare_dataset_loaders(self):

        # Set up dataset splits given the data splits
        overall_splits = defaultdict(lambda: defaultdict(dict))
        full_dataset = self.datasets["full"]
        for partition, splits in full_dataset.partition_splits.items():
            for split, split_idx in splits.items():
                overall_splits[split][partition] = split_idx

        full_dataset = self.datasets["full"]
        for split, partition_splits in overall_splits.items():
            self.datasets[split] = PartitionedSubset(full_dataset, partition_splits)

        self.dataloaders = {}
        self.dataloaders = {
            split: DataLoader(
                dataset,
                batch_sampler=PartitionedBatchSampler(
                    PartitionedDistributedSampler(
                        dataset,
                        num_replicas=self.num_replicas,
                        rank=self.rank,
                        shuffle=self.tasklist[None].shuffle if split == "train" else False,
                    ),
                    batch_size=self._cfg.trainer.per_device_batch_size,
                    drop_last=False,
                ),
                collate_fn=heimdall_collate_fn,
                # num_workers=4,  # TODO: currently doesn't work. To fix, will need to create
                # separate DataLoader for each partition, wrap them all with accelerate,
                # and return accordingly.
            )
            for split, dataset in self.datasets.items()
        }

        dataset_str = pformat(self.datasets).replace("\n", "\n\t")
        self.check_print(f"> Finished setting up datasets (and loaders):\n\t{dataset_str}", rank=True)
