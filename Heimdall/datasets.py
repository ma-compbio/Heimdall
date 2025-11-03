import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pprint import pformat
from typing import TYPE_CHECKING, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as PyTorchDataset
from torch.utils.data import Subset

from Heimdall.task import CellFeatType, LabelType, Task
from Heimdall.utils import MAIN_KEYS

if TYPE_CHECKING:
    from Heimdall.cell_representations import CellRepresentation

SPLIT_MASK_KEYS = {"full_mask", "train_mask", "val_mask", "test_mask", "full", "train", "val", "test"}


class Dataset(PyTorchDataset, ABC):
    SPLITS = ["train", "val", "test"]

    def __init__(self, data: "CellRepresentation", keys=MAIN_KEYS):
        super().__init__()
        self._data = data
        self.keys = keys

        self.splits = {}
        split_type = "predefined"
        self._setup_predefined_splits()  # predefined splits may be set up here

        # NOTE: need to setup labels first, index sizes might depend on it
        self._setup_idx()

        # Set up random splits if predefined splits are unavailable
        if not self.splits:
            split_type = "random"
            self._setup_random_splits()

        split_size_str = "\n  ".join(f"{i}: {len(j):,}" for i, j in self.splits.items())
        print(f"> Dataset splits sizes ({split_type}):\n  {split_size_str}")

    @property
    def idx(self) -> NDArray[np.int_]:
        return self.data._idx

    @property
    def data(self) -> "CellRepresentation":
        return self._data

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

    def _setup_random_splits(self):
        # warnings.warn("Pre-defined split unavailable, using random 6/2/2 split", UserWarning, stacklevel=2)

        size = len(self)
        seed = self.data._cfg.seed

        train_val_idx, test_idx = train_test_split(np.arange(size), train_size=0.6, random_state=seed)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=seed)

        self.splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    @abstractmethod
    def _setup_idx(self): ...

    @abstractmethod
    def _setup_predefined_splits(self, task: Task): ...

    @abstractmethod
    def get_shared_inputs(self, idx): ...

    def __getitem__(self, idx) -> Tuple[CellFeatType, LabelType]:
        shared_inputs = self.get_shared_inputs(idx)

        all_inputs = defaultdict(dict)
        for subtask_name, subtask in self.data.tasklist:
            subtask_inputs = subtask.get_inputs(idx, shared_inputs)
            for key in self.keys:
                default_value = shared_inputs[key] if key in shared_inputs else None
                subtask_input = subtask_inputs.get(key, default_value)
                all_inputs[key][subtask_name] = subtask_input

        return all_inputs


class SingleInstanceDataset(Dataset):
    def _setup_idx(self):
        self.data._idx = np.arange(self.data.adata.shape[0])

    def _setup_predefined_splits(self):
        adata = self.data.adata

        # Set up splits and task mask
        # splits = dataset_task_cfg.get("splits", None)
        if self.data.tasklist.splits is None or self.splits:
            return

        print("> Found predefined splits in config, extracting splits.")

        split_type = self.data.tasklist.splits.get("type", None)
        if split_type == "predefined":
            splits = {}
            if hasattr(self.data.tasklist.splits, "col"):
                split_col = adata.obs[self.data.tasklist.splits.col]
            else:
                split_col = adata.obs["split"]
            for split in self.SPLITS:
                if (split_key := self.data.tasklist.splits.keys_.get(split)) is None:
                    warnings.warn(
                        f"Skipping {split!r} split as the corresponding key is not found",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                splits[split] = np.where(split_col == split_key)[0]
            self.splits = splits
        else:
            raise ValueError(f"Unknown split type {split_type!r}")

    def get_shared_inputs(self, idx):
        identity_inputs, expression_inputs, expression_padding = self.data.fc[idx]

        return {
            "identity_inputs": identity_inputs,
            "expression_inputs": expression_inputs,
            "expression_padding": expression_padding,
        }


class PairedInstanceDataset(Dataset):
    def __init__(self, data: "CellRepresentation", keys=MAIN_KEYS):
        self._setup_obsp_task_keys(data)
        super().__init__(data, keys=keys)

    def _setup_idx(self):
        # NOTE: full mask is set up during runtime given split masks or the data
        mask = self.data.adata.obsp["full_mask"]
        self.data._idx = np.vstack(np.nonzero(mask)).T  # pairs x 2

    def _setup_obsp_task_keys(self, data: "CellRepresentation"):
        adata = data.adata

        all_obsp_task_keys, obsp_mask_keys = [], []
        for key in adata.obsp:
            (obsp_mask_keys if key in SPLIT_MASK_KEYS else all_obsp_task_keys).append(key)

        all_obsp_task_keys = sorted(all_obsp_task_keys)
        obsp_mask_keys = sorted(obsp_mask_keys)

        # Select task keys
        candidate_obsp_task_keys = data.tasklist.interaction_type
        if candidate_obsp_task_keys == "_all_":
            data.obsp_task_keys = all_obsp_task_keys
        else:
            # NOTE: in hydra, this can be either a list or a string
            if isinstance(candidate_obsp_task_keys, str):
                candidate_obsp_task_keys = [candidate_obsp_task_keys]

            if invalid_obsp_task_keys := [i for i in candidate_obsp_task_keys if i not in all_obsp_task_keys]:
                raise ValueError(
                    f"{len(invalid_obsp_task_keys)} out of {len(candidate_obsp_task_keys)} "
                    f"specified interaction types are invalid: {invalid_obsp_task_keys}\n"
                    f"Valid options are: {pformat(all_obsp_task_keys)}",
                )
            data.obsp_task_keys = candidate_obsp_task_keys

    def _setup_predefined_splits(self):
        adata = self.data.adata

        # Set up splits and task mask
        if self.data.tasklist.splits is None:  # no predefined splits specified
            full_mask = np.sum([np.abs(adata.obsp[i]) for i in self.data.obsp_task_keys], axis=-1) > 0
            nz = np.nonzero(full_mask)
        elif (split_type := self.data.tasklist.splits.type) == "predefined":
            print("> Found predefined splits in config, extracting splits.")
            masks = {}
            for split in self.SPLITS:
                if (split_key := self.data.tasklist.splits.keys_.get(split)) is None:
                    warnings.warn(
                        f"Skipping {split!r} split as the corresponding key is not found",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                masks[split] = adata.obsp[split_key]
            full_mask = np.sum(list(masks.values())).astype(bool)
            nz = np.nonzero(full_mask)

            # Set up predefined splits
            self.splits = {split: np.where(mask[nz])[1] for split, mask in masks.items()}

        else:
            raise ValueError(f"Unknown split type {split_type!r}")

        adata.obsp["full_mask"] = full_mask

    def get_shared_inputs(self, idx):
        identity_inputs, expression_inputs, expression_padding = zip(
            *[self.data.fc[cell_idx] for cell_idx in self.idx[idx]],
        )

        return {
            "identity_inputs": identity_inputs,
            "expression_inputs": expression_inputs,
            "expression_padding": expression_padding,
        }


class PartitionedDataset(SingleInstanceDataset):
    def __init__(self, data, *args, **kwargs):
        self.partition_splits = {}
        super().__init__(data, *args, **kwargs)

    @property
    def partition(self):
        return self._data.partition

    @property
    def num_partitions(self):
        return self._data.num_partitions

    @partition.setter
    def partition(self, partition):
        self._data.partition = partition
        self.splits = self.partition_splits.get(partition, None)

    @property
    def partition_sizes(self):
        return self._data.partition_sizes

    @property
    def num_cells(self):
        return self._data.num_cells

    def __len__(self):
        return self.partition_sizes[self.partition]

    def _setup_predefined_splits(self):
        splits = self.data.tasklist.splits

        if splits is None:
            return

        print("> Found predefined splits in config, extracting splits.")
        for partition in range(self.num_partitions):
            self.partition = partition
            self.partition_splits[partition] = self._get_partition_splits(partition)

        self.partition = 0

    def _setup_random_splits(self):
        print("> Did not find splits in config, generating random splits.")
        for partition in range(self.num_partitions):
            self.partition = partition
            self.partition_splits[partition] = self._get_random_splits_partition(
                partition,  # TODO: pass train_split correctly via self.data.tasklist.splits
            )

        self.partition = 0

    def _get_partition_splits(self, part_id):
        adata = self.data.adata

        partition_splits = {}
        if "col" in self.data.tasklist.splits:
            split_col = adata.obs[self.data.tasklist.splits.col]
        else:
            split_col = adata.obs["split"]

        for split in self.SPLITS:
            if (split_key := self.data.tasklist.splits.keys_.get(split)) is None:
                warnings.warn(
                    f"Skipping {split!r} split as the corresponding key is not found",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            partition_splits[split] = np.where(split_col == split_key)[0]

        return partition_splits

    def _get_random_splits_partition(self, part_id, train_split: float = 0.8):
        num_samples_partition = self.partition_sizes[part_id]

        # warnings.warn("Pre-defined split unavailable, using random split", UserWarning, stacklevel=2)

        seed = self._data._cfg.seed + part_id

        train_idx, test_val_idx = train_test_split(
            np.arange(num_samples_partition),
            train_size=train_split,
            random_state=seed,
        )
        val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, random_state=seed)

        return {"train": train_idx, "val": val_idx, "test": test_idx}


class PartitionedSubset(Subset):
    r"""Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        partition (sequence): Partition indices in the whole set selected for subset

    """

    def __init__(self, dataset: PartitionedDataset, indices: dict[list]) -> None:
        self.dataset = dataset
        self._indices = indices

    @property
    def indices(self):
        return self._indices[self.dataset.partition]

    # def __getitem__(self, idx):
    #     if isinstance(idx, list):
    #         return self.__getitems__(idx)

    #     # index, partition = idx
    #     # if partition != self.dataset.partition:
    #     #     self.dataset.partition = partition
    #     return self.dataset[self.indices[idx]]

    # def __getitems__(self, indices: list[tuple[int, int]]) -> list:
    #     # add batched sampling support when parent dataset supports it.
    #     # see torch.utils.data._utils.fetch._MapDatasetFetcher
    #     if callable(getattr(self.dataset, "__getitems__", None)):
    #         return self.dataset.__getitems__([self.indices[idx] for idx in indices])
    #     else:
    #         batched_data = []
    #         # print(f'{indices=}')
    #         # print(f'{len(self.indices[self.dataset.partition])=}')
    #         for idx in indices:
    #             print(f'{idx=}')
    #             print(f'{len(self.indices[self.dataset.partition])=}')
    #             batched_data.append(self.dataset[self.indices[idx]])

    #         return batched_data

    def __len__(self):
        return sum(len(indices) for indices in self._indices.values())


# class PartitionedDataLoader:
#     """Custom DataLoader that handles multiple partitions and raises custom
#     exceptions."""
#
#     def __init__(self, dataset: PartitionedSubset | PartitionedDataset, **dataloader_kwargs):
#         """
#         Args:
#             **dataloader_kwargs: Additional arguments for DataLoader
#         """
#         self.dataloader_kwargs = dataloader_kwargs
#         self.shuffle = self.dataloader_kwargs.get("shuffle", False)
#         self.dataset = dataset
#         if isinstance(self.dataset, Subset):
#             subset = dataset
#             self.full_dataset = subset.dataset
#         else:
#             self.full_dataset = self.dataset
#
#         self.partition_order = list(range(self.num_partitions))
#         self.partition_idx = None
#
#     @property
#     def partition_idx(self):
#         return self._partition_idx
#
#     @property
#     def num_partitions(self):
#         return self.full_dataset.num_partitions
#
#     @partition_idx.setter
#     def partition_idx(self, partition_idx: int | None):
#         self._partition_idx = partition_idx
#         if partition_idx is None:
#             return
#
#         partition = self.partition_order[partition_idx]
#
#         # load underlying partition
#         self.full_dataset.partition = partition
#
#         # create dataloader for partition
#         self.dataloader = DataLoader(
#             self.dataset,
#             **self.dataloader_kwargs,
#         )
#         self.iterator = iter(self.dataloader)  # TODO: Is this necessary? Isn't DataLoader already an iterator?
#
#     def __iter__(self):
#         if self.shuffle:
#             self.partition_order = np.random.shuffle(self.partition_order)
#
#         if self.partition_idx is None:
#             self.partition_idx = 0
#
#         return self
#
#     def __next__(self):
#         try:
#             result =  next(self.iterator)
#             print(result.keys())
#             return result
#         except StopIteration:
#             self.full_dataset.data.accelerator.wait_for_everyone()
#             if self.partition_idx + 1 == self.num_partitions:
#                 self.partition_idx = None
#                 raise AllPartitionsExhausted()
#             else:
#                 self.partition_idx += 1
#                 return next(self.iterator)
#
#     def __len__(self):
#         return len(self.dataloader_kwargs["sampler"]) // self.full_dataset.data._cfg.trainer.per_device_batch_size
