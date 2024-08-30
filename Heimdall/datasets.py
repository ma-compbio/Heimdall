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

        # TODO: specify task type multiclass/multilabel/regression in config
        if len(obsp_task_keys) == 1:
            task_mat = adata.obsp[obsp_task_keys[0]]
            assert (task_mat.data > 0).all(), "Multiclass task id must be positive"

            num_tasks = task_mat.max()  # class id starts from 1. 0's are ignoreed
            labels = np.array(task_mat[nz]).ravel().astype(np.int64) - 1  # class 0 is not used
        else:
            num_tasks = len(obsp_task_keys)

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


class PretrainDataset(SingleInstanceDataset, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_labels(self):
        # FIX: not necessarily the case,e.g., UCE.....
        self.labels = self.data.cell_representations.copy()

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
        self._num_tasks = self.data.sequence_length  # number of genes

    @property
    def mask_token(self):
        return self.data.special_tokens["mask"]

    def _transform(self, data):
        size = data["labels"].size
        mask = np.random.random(size) < self.mask_ratio

        # Ignore padding tokens
        is_padding = data["labels"] == self.data.special_tokens["pad"]
        mask[is_padding] = False

        data["inputs"][mask] = self.mask_token
        data["masks"] = mask

        return data
