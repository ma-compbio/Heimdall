"""The Cell Representation Object for Processing."""

import pickle as pkl
import textwrap
from collections import defaultdict
from functools import partial, wraps
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, Optional, Union

import anndata as ad
import numpy as np
import scanpy as sc
from accelerate import Accelerator, DistributedDataParallelKwargs
from numpy.typing import NDArray
from omegaconf import OmegaConf, open_dict
from scipy import sparse
from scipy.sparse import csc_array
from torch.utils.data import DataLoader, Subset

from Heimdall.datasets import Dataset, PartitionedSubset
from Heimdall.fc import Fc
from Heimdall.fe import Fe
from Heimdall.fg import Fg
from Heimdall.samplers import PartitionedBatchSampler, PartitionedDistributedSampler
from Heimdall.task import Tasklist

# from Heimdall.samplers import PartitionedDistributedSampler
from Heimdall.utils import (
    conditional_print,
    convert_to_ensembl_ids,
    get_collation_closure,
    get_fully_qualified_cache_paths,
    get_value,
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
            assert self.adata is not None, "no adata found, Make sure to run load_anndata() first"

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
        self.special_tokens = {token: self.num_genes + i for i, token in enumerate(self._SPECIAL_TOKENS)}


class CellRepresentation(SpecialTokenMixin):
    TOKENIZER_KEYS = ("fg", "fe", "fc")
    DATASET_KEYS = ("dataset.preprocess_args.data_path",)

    def __init__(self, config, accelerator: Accelerator, auto_setup: bool = True):
        """Initialize the Cell Rep object with configuration and AnnData object.

        Parameters:
        config (dict): Configuration dictionary.

        """
        self.rank = accelerator.process_index
        self.num_replicas = accelerator.num_processes
        self.accelerator = accelerator
        self._indent = ""
        self._save_precomputed = False
        self._get_precomputed = False

        self.setup_finished = False
        self._cfg = config

        self.dataset_preproc_cfg = config.dataset.preprocess_args

        self.fg_cfg = config.fg
        self.fc_cfg = config.fc
        self.fe_cfg = config.fe
        self.float_dtype = config.float_dtype
        self.adata = None
        self.processed_fcfg = False
        self.verbose = 0  # TODO: expose

        seed = 0  # TODO: make this configurable???
        self.rng = np.random.default_rng(seed)

        if auto_setup:
            self.create_tasklist()
            self.setup(setup_labels=False)
            SpecialTokenMixin.__init__(self)
            self.prepare_full_dataset()
            self.setup_labels()
            self.setup_finished = True
            self.prepare_dataset_loaders()

    @property
    def indent(self):
        return self._indent

    @indent.setter
    def indent(self, val: int):
        self._indent = " " * (val * 4)

    @property
    def save_precomputed(self):
        return self._save_precomputed

    @property
    def get_precomputed(self):
        return self._get_precomputed

    def setup_labels(self, hash_vars=()):
        """Can only be called after `self.adata` and `self.datasets` is
        populated."""

        if not hasattr(self, "datasets"):
            return

        for subtask_name, subtask in self.tasklist:
            if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
                cache_dir = Path(cache_dir)
                is_cached = subtask.from_cache(cache_dir, hash_vars=hash_vars, task_name=subtask_name)
                if is_cached:
                    continue
            subtask.setup_labels()
            if cache_dir is not None:
                subtask.to_cache(cache_dir, hash_vars=hash_vars, task_name=subtask_name)

        self.print_during_setup("> Finished setting up labels", is_printable_process=True)

    def setup(self, hash_vars=(), setup_labels=True):
        self.load_anndata()
        self.setup_tokenizer(hash_vars=hash_vars)
        if setup_labels:
            self.setup_labels(hash_vars=hash_vars)
        # if hasattr(self, "datasets") and "full" in self.datasets:
        #     self.prepare_dataset_loaders()

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

    @property
    def gene_names(self, mask_key: str = "identity_valid_mask"):
        if hasattr(self, "_gene_names"):
            return self._gene_names

        if mask_key in self.adata.var:
            valid_mask = self.adata.var[mask_key]
            self._gene_names = self.adata.var_names[valid_mask]

            return self._gene_names

        return self.adata.var_names

    @property
    def num_genes(self):
        return len(self.gene_names)

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

        _, gene_mapping = convert_to_ensembl_ids(self.adata, data_dir, species=species, verbose=not self.setup_finished)
        return self.adata, gene_mapping

    def anndata_from_cache(self, preprocessed_data_path, preprocessed_cfg_path, cfg):
        if preprocessed_data_path.is_file():
            self.print_during_setup(
                f"> Found already preprocessed anndata: {preprocessed_data_path}",
                is_printable_process=True,
            )
            # loaded_cfg_str = OmegaConf.to_yaml(OmegaConf.load(preprocessed_cfg_path)).replace("\n", "\n    ")
            # print(f"  Preprocessing config:\n    {loaded_cfg_str}") # TODO: add verbosity level
            self.adata = ad.read_h5ad(
                preprocessed_data_path,
                backed="r",
            )  # add backed argument to prevent entire dataset from being read into mem
            return True

        # OmegaConf.save(cfg, preprocessed_cfg_path)

        return False

    def anndata_to_cache(self, preprocessed_data_path):
        if preprocessed_data_path is not None:
            self.print_during_setup("> Writing preprocessed Anndata Object", is_printable_process=True)
            self.adata.write(preprocessed_data_path)
            self.print_during_setup("> Finished writing preprocessed Anndata Object", is_printable_process=True)

    def create_tasklist(self):
        if hasattr(self._cfg.tasks.args, "subtask_configs"):
            self.tasklist = instantiate_from_config(self._cfg.tasks, self)
        else:
            self.tasklist = Tasklist(
                self,
                subtask_configs={"default": self._cfg.tasks},
                dataset_config=self._cfg.tasks.args.dataset_config,
            )

        self.num_subtasks = self.tasklist.num_subtasks

    def load_anndata(self, filename: str = "data.h5ad"):
        """Load AnnData into memory (and preprocess, if necessary)."""
        if self.adata is not None:
            raise ValueError("Anndata object already exists, are you sure you want to reprocess again?")

        keys = self.DATASET_KEYS
        preprocessed_data_path = preprocessed_cfg_path = cfg = None
        if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
            cache_dir = Path(cache_dir)
            preprocessed_data_path, preprocessed_cfg_path, cfg = get_fully_qualified_cache_paths(
                self._cfg,
                cache_dir / "processed_anndata",
                filename,
                keys=keys,
            )
            is_cached = self.anndata_from_cache(preprocessed_data_path, preprocessed_cfg_path, cfg)
            if is_cached:
                self.print_during_setup(f"> Finished loading AnnData with shape: {self.adata.shape}")
                return

        self.preprocess_anndata()
        self.anndata_to_cache(preprocessed_data_path)

        self.print_during_setup(f"> Finished loading AnnData with shape: {self.adata.shape}")

    def preprocess_anndata(self):
        self.adata = ad.read_h5ad(self.dataset_preproc_cfg.data_path)
        self.print_during_setup(f"> Finished Loading in {self.dataset_preproc_cfg.data_path}")
        # convert gene names to ensembl ids
        self.print_during_setup("> Converting gene names to Ensembl IDs...")
        self.adata, _ = self.convert_to_ensembl_ids(
            data_dir=self._cfg.ensembl_dir,
            species=self.dataset_preproc_cfg.species,
        )

        if get_value(self.dataset_preproc_cfg, "normalize"):
            self.print_during_setup("> Normalizing AnnData...")

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
            self.print_during_setup("> Skipping Normalizing anndata...")

        if get_value(self.dataset_preproc_cfg, "log_1p"):
            self.print_during_setup("> Log Transforming anndata...")

            sc.pp.log1p(self.adata)
        else:
            self.print_during_setup("> Skipping Log Transforming anndata..")

        if get_value(self.dataset_preproc_cfg, "top_n_genes") and self.dataset_preproc_cfg["top_n_genes"] != "false":
            # Identify highly variable genes
            self.print_during_setup(
                f"> Using highly variable subset... top {self.dataset_preproc_cfg.top_n_genes} genes",
            )
            sc.pp.highly_variable_genes(self.adata, n_top_genes=self.dataset_preproc_cfg.top_n_genes)
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()
        else:
            self.print_during_setup("> No highly variable subset... using entire dataset")

        if get_value(self.dataset_preproc_cfg, "scale_data"):
            # Scale the data
            raise NotImplementedError("Scaling the data is NOT RECOMMENDED, please set it to false")
            self.print_during_setup("> Scaling the data...")
            sc.pp.scale(self.adata, max_value=10)
        else:
            self.print_during_setup("> Not scaling the data...")

        if get_value(self.dataset_preproc_cfg, "get_medians"):
            # Get medians
            self.print_during_setup("> Getting nonzero medians...")
            csc_expression = csc_array(self.adata.X)
            genewise_nonzero_expression = np.split(csc_expression.data, csc_expression.indptr[1:-1])
            gene_medians = np.array([np.median(gene_nonzeros) for gene_nonzeros in genewise_nonzero_expression])
            self.adata.var["medians"] = gene_medians

        self.print_during_setup(f"> Finished processing AnnData object:\n{self.adata}")

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
        heimdall_collate_fn = get_collation_closure()
        dataloader_kwargs = {"num_workers": 4}  # TODO: we can parse additional data loader kwargs from config
        per_device_batch_size = self.tasklist.batchsize // self.accelerator.num_processes
        self.dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=per_device_batch_size,
                shuffle=self.tasklist.shuffle if split == "train" else False,
                collate_fn=heimdall_collate_fn,
                **dataloader_kwargs,
            )
            for split, dataset in self.datasets.items()
        }

        dataset_str = pformat(self.datasets).replace("\n", "\n\t")
        self.print_during_setup(
            f"> Finished setting up datasets (and loaders):\n\t{dataset_str}",
            is_printable_process=True,
        )

    def get_tokenizer_cache_path(self, cache_dir, hash_vars, filename: str = "data.pkl"):
        keys = set(self.DATASET_KEYS).union(set(self.TOKENIZER_KEYS))

        processed_data_path, _, _ = get_fully_qualified_cache_paths(
            self._cfg,
            cache_dir / "processed_data",
            filename,
            keys=keys,
            hash_vars=hash_vars,
        )

        return processed_data_path

    def load_tokenizer_from_cache(self, cache_dir, hash_vars):
        processed_data_path = self.get_tokenizer_cache_path(cache_dir, hash_vars)
        if processed_data_path.is_file():
            self.print_during_setup(
                f"> Found already processed `CellRepresentation`: {processed_data_path}",
                is_printable_process=True,
            )
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

        # TODO: add back
        # OmegaConf.save(cfg, processed_cfg_path)
        return False

    def save_tokenizer_to_cache(self, cache_dir, hash_vars):
        # Gather things for caching
        identity_embedding_index, identity_valid_mask = self.fg.__getitem__(self.adata.var_names, return_mask=True)

        gene_embeddings = self.fg.gene_embeddings
        expression_embeddings = self.fe.expression_embeddings

        processed_data_path = self.get_tokenizer_cache_path(cache_dir, hash_vars)
        if not processed_data_path.is_file():
            with open(processed_data_path, "wb") as rep_file:
                cache_representation = (
                    identity_embedding_index,
                    identity_valid_mask,
                    gene_embeddings,
                    expression_embeddings,
                )
                pkl.dump(cache_representation, rep_file)
                self.print_during_setup(f"> Finished writing cell representations at {processed_data_path}")

    def instantiate_representation_functions(self):
        """Instantiate `f_g`, `fe` and `f_c` according to config."""
        self.fg: Fg
        self.fe: Fe
        self.fc: Fc
        self.fg, fg_name = instantiate_from_config(
            self.fg_cfg,
            self,
            vocab_size=self.adata.n_vars + 2,
            rng=self.rng,
            return_name=True,
        )
        self.fe, fe_name = instantiate_from_config(
            self.fe_cfg,
            self,
            vocab_size=self.adata.n_vars + 2,  # TODO: figure out a way to fix the number of expr tokens
            rng=self.rng,
            return_name=True,
        )
        self.fc, fc_name = instantiate_from_config(
            self.fc_cfg,
            self.fg,
            self.fe,
            self,
            float_dtype=self.float_dtype,
            rng=self.rng,
            return_name=True,
        )

    @check_states(adata=True)
    def setup_tokenizer(self, hash_vars=()):
        """Processes the `f_g`, `fe` and `f_c` from the config.

        This will first check to see if the cell representations are already
        cached, and then will either load the cached representations or compute
        them and save them.

        """

        self.instantiate_representation_functions()
        if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
            cache_dir = Path(cache_dir)
            is_cached = self.load_tokenizer_from_cache(cache_dir, hash_vars=hash_vars)
            if is_cached:
                return

        self.fg.preprocess_embeddings()
        self.print_during_setup(f"> Finished calculating fg with {self.fg_cfg.type}")

        self.fe.preprocess_embeddings()
        self.print_during_setup(f"> Finished calculating fe with {self.fe_cfg.type}")

        self.processed_fcfg = True

        if cache_dir is not None:
            self.save_tokenizer_to_cache(cache_dir, hash_vars=hash_vars)

    def print_r0(self, message):
        conditional_print(f"{message}", self.accelerator.is_main_process)

    def print_during_setup(self, message, is_printable_process=False):
        message = textwrap.indent(message, self.indent)
        # message = self.indent + message
        if not self.setup_finished:
            if is_printable_process:
                print(message)
            else:
                self.print_r0(message)


class PartitionedCellRepresentation(CellRepresentation):
    def __init__(self, config, accelerator: Accelerator, auto_setup: bool = True):
        super().__init__(config, accelerator, auto_setup=False)

        self.dataset_directory = str(self.dataset_preproc_cfg.data_path)

        # Expect `data_path` to hold parent directory, not filepath
        self.partition_file_paths = sorted(
            Path(self.dataset_preproc_cfg.data_path).glob("*.h5ad"),
        )
        self.partition_folder = str(self.dataset_preproc_cfg.data_path)
        self.num_partitions = len(self.partition_file_paths)

        if self.num_partitions == 0:
            raise ValueError(
                "No partitions were found under the directory at "
                f"'{self.dataset_preproc_cfg.data_path}'. The dataset path "
                "(`config.dataset.preprocess_args.data_path`) is probably set incorrectly.",
            )

        self.partition_sizes = {}
        self.num_cells = {}
        if auto_setup:
            self.create_tasklist()

            self.print_during_setup("> Setting up partition_sizes...")
            self.indent = 1
            if accelerator.is_main_process:  # One time through for main process
                self.setup(setup_labels=False)

            accelerator.wait_for_everyone()
            if not accelerator.is_main_process:  # Let others catch up (just for `set_partition_size`)
                self.setup_finished = True
                self.setup(setup_labels=False)
            self.indent = 0

            self.prepare_full_dataset()  # Setup dataset before preparing labels
            self.print_during_setup("> Setting up labels...")
            self.indent = 1
            self.setup(setup_labels=True)
            self.indent = 0
            self.setup_finished = True

            self.prepare_dataset_loaders()

            SpecialTokenMixin.__init__(self)  # TODO: this works because all partitions have the
            # same `var_names`. Can we enforce that during preprocessing?

    def set_partition_size(self):
        """Get the size of the current partition."""
        self.partition_sizes[self.partition] = self.adata.n_obs
        self.num_cells[self.partition] = self.adata.n_obs

    def setup(self, setup_labels=False):
        for partition in range(self.num_partitions):  # Setting up AnnData and sizes
            self.prepare_partition(partition)

            self.indent = 2
            super().setup(hash_vars=(int(self.partition),), setup_labels=setup_labels)
            self.indent = 1

            self.set_partition_size()

    def preprocess_anndata(self):
        self.dataset_preproc_cfg.data_path = self.partition_file_paths[self.partition]
        super().preprocess_anndata()
        self.dataset_preproc_cfg.data_path = self.partition_folder

    def load_anndata(self, filename="data.h5ad"):
        partition_filename = f"partition_{self.partition}_{filename}"
        super().load_anndata(partition_filename)

    def close_partition(self):
        """Close current partition."""
        if self.adata is not None:
            self.adata.file.close()

            self.print_during_setup(
                f"> Closing partition {self.partition + 1} of {self.num_partitions}",
            )
            if self.save_precomputed:
                self.adata.write_h5ad(self.adata.filename)

            del self.adata
            self.adata = None

    @property
    def partition(self):
        return self._partition

    def prepare_partition(self, partition):
        self.close_partition()
        self._partition = partition
        if partition is not None:
            self.print_during_setup(
                f"> Opening partition {partition + 1} of {self.num_partitions}",
            )
            return True

        return False

    @partition.setter
    def partition(self, partition):
        """Move to a new partition."""
        if getattr(self, "_partition", None) == partition:
            return

        # Preprocess partition AnnData
        partition_prepared = self.prepare_partition(partition)
        if partition_prepared:
            super().setup(hash_vars=(int(self.partition),), setup_labels=True)

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
        heimdall_collate_fn = get_collation_closure()
        per_device_batch_size = self.tasklist.batchsize // self.accelerator.num_processes
        self.dataloaders = {
            split: DataLoader(
                dataset,
                batch_sampler=PartitionedBatchSampler(
                    PartitionedDistributedSampler(
                        dataset,
                        num_replicas=self.num_replicas,
                        rank=self.rank,
                        shuffle=self.tasklist.shuffle if split == "train" else False,
                    ),
                    batch_size=per_device_batch_size,
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
        self.print_during_setup(
            f"> Finished setting up datasets (and loaders):\n\t{dataset_str}",
            is_printable_process=True,
        )


def setup_accelerator(config, cpu=False, run_wandb=False):
    # get accelerate context
    accelerator_log_kwargs = {}
    if run_wandb:
        accelerator_log_kwargs["log_with"] = "wandb"
        accelerator_log_kwargs["project_dir"] = config.work_dir

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.trainer.args.accumulate_grad_batches,
        step_scheduler_with_optimizer=False,
        cpu=cpu,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
        **accelerator_log_kwargs,
    )

    return accelerator


def setup_data(config, cpu=False, accelerator=None):
    """Set up Heimdall data based on config, including cr and accelerator."""

    run_wandb = getattr(config, "run_wandb", False)
    if accelerator is None:
        accelerator = setup_accelerator(config, cpu=cpu, run_wandb=run_wandb)

    if accelerator.is_main_process:
        print(OmegaConf.to_yaml(config, resolve=True))

    with open_dict(config):
        only_preprocess_data = config.pop("only_preprocess_data", None)
        # pop so hash of cfg is not changed depending on value

    cr = instantiate_from_config(config.tasks.cell_rep_config, config, accelerator)

    return accelerator, cr, run_wandb, only_preprocess_data
