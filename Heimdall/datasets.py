import warnings
from abc import ABC, abstractmethod, abstractproperty
from pprint import pformat
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as PyTorchDataset

if TYPE_CHECKING:
    from Heimdall.cell_representations import CellRepresentation

SPLIT_MASK_KEYS = {"full_mask", "train_mask", "val_mask", "test_mask", "full", "train", "val", "test"}

CellFeatType = Union[NDArray[np.int_], NDArray[np.float32]]
FeatType = Union[CellFeatType, Tuple[CellFeatType, CellFeatType]]
LabelType = Union[NDArray[np.int_], NDArray[np.float32]]


class Dataset(PyTorchDataset, ABC):
    SPLITS = ["train", "val", "test"]

    def __init__(self, data: "CellRepresentation"):
        super().__init__()
        self._data = data

        if self.labels is None:
            self._setup_labels_and_pre_splits()  # predefined splits may be set up here

        # NOTE: need to setup labels first, index sizes might depend on it
        self._setup_idx()

        # Set up random splits if predefined splits are unavailable
        split_type = "predefined"
        if self.splits is None:
            self._setup_random_splits()
            split_type = "random"
        split_size_str = "\n  ".join(f"{i}: {len(j):,}" for i, j in self.splits.items())
        print(f"> Dataset splits sizes ({split_type}):\n  {split_size_str}")

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

    def _setup_random_splits(self):
        warnings.warn("Pre-defined split unavailable, using random 6/2/2 split", UserWarning, stacklevel=2)

        size = len(self)
        seed = self.data._cfg.seed

        train_val_idx, test_idx = train_test_split(np.arange(size), train_size=0.6, random_state=seed)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=seed)

        self.splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    @abstractmethod
    def _setup_idx(self): ...

    @abstractmethod
    def _setup_labels_and_pre_splits(self): ...

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[FeatType, LabelType]: ...


def filter_list(input_list):
    keywords = ["train", "test", "val"]
    return [item for item in input_list if any(keyword in item.lower() for keyword in keywords)]


class SingleInstanceDataset(Dataset):
    def _setup_idx(self):
        self._idx = np.arange(self.data.adata.shape[0])

    def _setup_labels_and_pre_splits(self):
        adata = self.data.adata
        dataset_task_cfg = self.data.dataset_task_cfg

        if "label_col_name" in dataset_task_cfg:
            assert "label_obsm_name" not in dataset_task_cfg
            df = adata.obs
            class_mapping = {
                label: idx
                for idx, label in enumerate(
                    df[dataset_task_cfg.label_col_name].unique(),
                    start=0,
                )
            }
            df["class_id"] = df[dataset_task_cfg.label_col_name].map(class_mapping)
            labels = np.array(df["class_id"])
            if dataset_task_cfg.task_type == "regression":
                labels = labels.reshape(-1, 1).astype(np.float32)

        elif "label_obsm_name" in dataset_task_cfg:
            assert "label_col_name" not in dataset_task_cfg
            df = adata.obsm[dataset_task_cfg.label_obsm_name]

            if dataset_task_cfg.task_type == "binary":
                (labels := np.empty(df.shape, dtype=np.float32)).fill(np.nan)
                labels[np.where(df == 1)] = 1
                labels[np.where(df == -1)] = 0
            elif dataset_task_cfg.task_type == "regression":
                labels = np.array(df).astype(np.float32)

            print(f"labels shape {labels.shape}")

        else:
            raise ValueError("Either 'label_col_name' or 'label_obsm_name' needs to be set.")
        self.labels = labels

        # Set up splits and task mask
        if "splits" not in dataset_task_cfg:  # no predefined splits specified
            pass

        elif (split_type := dataset_task_cfg.splits.type) == "predefined":
            self.splits = {}
            if hasattr(dataset_task_cfg.splits, "col"):
                split_col = adata.obs[dataset_task_cfg.splits.col]
            else:
                split_col = adata.obs["split"]
            for split in self.SPLITS:
                if (split_key := dataset_task_cfg.splits.keys_.get(split)) is None:
                    warnings.warn(
                        f"Skipping {split!r} split as the corresponding key is not found",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                self.splits[split] = np.where(split_col == split_key)[0]

        else:
            raise ValueError(f"Unknown split type {split_type!r}")

    def __getitem__(self, idx) -> Tuple[CellFeatType, LabelType]:
        identity_inputs, expression_inputs, expression_padding = self.data.fc[idx]

        return {
            "identity_inputs": identity_inputs,
            "expression_inputs": expression_inputs,
            "expression_padding": expression_padding,
            "labels": self.data.labels[idx],
        }


class PairedInstanceDataset(Dataset):
    def _setup_idx(self):
        # NOTE: full mask is set up during runtime given split masks or the data
        mask = self.data.adata.obsp["full_mask"]
        self._idx = np.vstack(np.nonzero(mask)).T  # pairs x 2

    def _setup_labels_and_pre_splits(self):
        adata = self.data.adata
        dataset_task_cfg = self.data.dataset_task_cfg

        all_obsp_task_keys, obsp_mask_keys = [], []
        for key in adata.obsp:
            (obsp_mask_keys if key in SPLIT_MASK_KEYS else all_obsp_task_keys).append(key)

        all_obsp_task_keys = sorted(all_obsp_task_keys)
        obsp_mask_keys = sorted(obsp_mask_keys)

        # Select task keys
        candidate_obsp_task_keys = dataset_task_cfg.interaction_type
        if candidate_obsp_task_keys == "_all_":
            obsp_task_keys = all_obsp_task_keys
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
            obsp_task_keys = candidate_obsp_task_keys

        # Set up splits and task mask
        if "splits" not in dataset_task_cfg:  # no predefined splits specified
            full_mask = np.sum([np.abs(adata.obsp[i]) for i in obsp_task_keys], axis=-1) > 0
            nz = np.nonzero(full_mask)

        elif (split_type := dataset_task_cfg.splits.type) == "predefined":
            masks = {}
            for split in self.SPLITS:
                if (split_key := dataset_task_cfg.splits.keys_.get(split)) is None:
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

        # Task type specific handling
        task_type = dataset_task_cfg.task_type
        if task_type == "multiclass":
            if len(obsp_task_keys) > 1:
                raise ValueError(f"{task_type!r} only supports a single task key, provided task keys: {obsp_task_keys}")

            task_mat = adata.obsp[obsp_task_keys[0]]
            num_tasks = task_mat.max()  # class id starts from 1. 0's are ignoreed
            labels = np.array(task_mat[nz]).ravel().astype(np.int64) - 1  # class 0 is not used

        elif task_type == "binary":
            num_tasks = len(obsp_task_keys)

            (labels := np.empty((len(nz[0]), num_tasks), dtype=np.float32)).fill(np.nan)
            for i, task in enumerate(obsp_task_keys):
                label_i = np.array(adata.obsp[task][nz]).ravel()
                labels[:, i][label_i == 1] = 1
                labels[:, i][label_i == -1] = 0

        elif task_type == "regression":
            num_tasks = len(obsp_task_keys)

            labels = np.zeros((len(nz[0]), num_tasks), dtype=np.float32)
            for i, task in enumerate(obsp_task_keys):
                labels[:, i] = np.array(adata.obsp[task][nz]).ravel()

        else:
            raise ValueError(f"task_type must be one of: 'multiclass', 'binary', 'regression'. Got: {task_type!r}")

        self.labels = labels

    def __getitem__(self, idx) -> Tuple[Tuple[CellFeatType, CellFeatType], LabelType]:
        identity_inputs, expression_inputs, expression_padding = zip(
            *[self.data.fc[cell_idx] for cell_idx in self.idx[idx]],
        )

        return {
            "identity_inputs": identity_inputs,
            "expression_inputs": expression_inputs,
            "expression_padding": expression_padding,
            "labels": self.data.labels[idx],
        }


class PretrainDataset(SingleInstanceDataset, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_labels_and_pre_splits(self):
        # # FIX: not necessarily the case,e.g., UCE.....
        # # FIX: probably doesn't work after we changed fg/fe/fc implementation...
        # identity_inputs, expression_inputs = self.data.fc[:]
        identity_inputs = [self.data.fc[i][0] for i in range(len(self.data.adata))]
        identity_inputs = np.vstack(identity_inputs).astype(int)
        self.labels = identity_inputs
        # self.labels = self.data.fc.copy()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self._transform(data)

    @abstractmethod
    def _transform(self, data): ...


class MaskedPretrainDataset(PretrainDataset, ABC):
    def __init__(self, *args, mask_ratio: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio

    @abstractproperty
    def mask_token(self): ...


class SeqMaskedPretrainDataset(MaskedPretrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_tasks = self.data.adata.n_vars  # number of genes

    @property
    def mask_token(self):
        return self.data.special_tokens["mask"]

    def _transform(self, data):
        size = data["labels"].size
        mask = np.random.random(size) < self.mask_ratio

        # Ignore padding tokens
        is_padding = data["labels"] == self.data.special_tokens["pad"]
        mask[is_padding] = False

        data["identity_inputs"][mask] = self.mask_token
        # data["expression_inputs"][mask] = self.mask_token
        data["masks"] = mask

        return data
