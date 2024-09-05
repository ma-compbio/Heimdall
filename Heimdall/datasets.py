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
        self.predefined_split = None

        # we can set up the specific splits here
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
        size = len(self)
        seed = self.data._cfg.seed

        if self.predefined_split is None:
            warnings.warn(
                "Pre-defined split unavailable, using random 6/2/2 split",
                UserWarning,
                stacklevel=2,
            )
            train_val_idx, test_idx = train_test_split(np.arange(size), train_size=0.6, random_state=seed)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=seed)

        else:
            warnings.warn(
                "Using Predefined Split",
                UserWarning,
                stacklevel=2,
            )

            # must be train test val, in that order
            all_idx = np.arange(size)
            train_len = len(self.labels["train"])
            val_len = len(self.labels["val"])
            test_len = len(self.labels["test"])
            train_idx = all_idx[:train_len]
            test_idx = all_idx[train_len:test_len]
            val_idx = all_idx[test_len:val_len]

        self.splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    @abstractmethod
    def _setup_idx(self): ...

    @abstractmethod
    def _setup_labels(self): ...

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[FeatType, LabelType]: ...


def filter_list(input_list):
    keywords = ["train", "test", "val"]
    return [item for item in input_list if any(keyword in item.lower() for keyword in keywords)]


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

        if self.predefined_split is None:

            mask = self.data.adata.obsp["full_mask"]
            self._idx = np.vstack(np.nonzero(mask)).T  # pairs x 2
        else:
            # we already have the masks defined
            # we can and will just vstack and concatenate the three together into one big set
            train_mask = self.data.adata.obsp["train"]
            test_mask = self.data.adata.obsp["test"]
            val_mask = self.data.adata.obsp["val"]

            train_idx = np.vstack(np.nonzero(train_mask)).T  # pairs x 2
            test_idx = np.vstack(np.nonzero(test_mask)).T  # pairs x 2
            val_idx = np.vstack(np.nonzero(val_mask)).T  # pairs x 2

            # adding it together into one large dataset, but the order of train/test/val is preserved
            self._idx = np.concatenate([train_idx, test_idx, val_idx], axis=0)

    def _setup_labels(self):
        adata = self.data.adata
        dataset_task_cfg = self.data.dataset_task_cfg

        all_obsp_task_keys, obsp_mask_keys = [], []
        for key in adata.obsp:
            (obsp_mask_keys if key in SPLIT_MASK_KEYS else all_obsp_task_keys).append(key)

        all_obsp_task_keys = sorted(all_obsp_task_keys)
        obsp_mask_keys = sorted(obsp_mask_keys)

        # in hydra, this can be either a list or a string
        candidate_obsp_task_keys = dataset_task_cfg.interaction_type
        if isinstance(candidate_obsp_task_keys, str):
            candidate_obsp_task_keys = [candidate_obsp_task_keys]

        if invalid_obsp_task_keys := [i for i in candidate_obsp_task_keys if i not in all_obsp_task_keys]:
            raise ValueError(
                f"{len(invalid_obsp_task_keys)} out of {len(candidate_obsp_task_keys)} "
                f"specified interaction types are invalid: {invalid_obsp_task_keys}\n"
                f"Valid options are: {pformat(all_obsp_task_keys)}",
            )
        obsp_task_keys = candidate_obsp_task_keys

        task_type = dataset_task_cfg.task_type
        if task_type == "multiclass":

            # this should only be the single interaction obsp of interest
            assert len(obsp_task_keys) == 1
            # not using a predefined split
            if dataset_task_cfg.splits == "all":
                full_mask = np.sum([np.abs(adata.obsp[i]) for i in obsp_task_keys], axis=-1) > 0
                adata.obsp["full_mask"] = full_mask
                nz = np.nonzero(full_mask)

                task_mat = adata.obsp[obsp_task_keys[0]]
                assert (task_mat.data > 0).all(), "Multiclass task id must be positive"

                num_tasks = task_mat.max()  # class id starts from 1. 0's are ignoreed
                labels = np.array(task_mat[nz]).ravel().astype(np.int64) - 1  # class 0 is not used
                self.labels = labels

            elif dataset_task_cfg.splits == "predefined":
                self.predefined_split = True
                task_mat = adata.obsp[obsp_task_keys[0]]  # same taskmat for everything
                assert (task_mat.data > 0).all(), "Multiclass task id must be positive"

                labels = {}
                for split in ["train", "test", "val"]:
                    split_mask_name = next((key for key in obsp_mask_keys if split in key), None)
                    if split_mask_name is None:
                        raise ValueError(f"No key found for split '{split}' in obsp_task_keys")

                    split_mask = adata.obsp[split_mask_name]
                    split_nz = np.nonzero(split_mask)
                    split_labels = np.array(task_mat[split_nz]).ravel().astype(np.int64) - 1  # class 0 is not used
                    labels[split] = split_labels

                    # renaming the masks for conveniences here
                    if split != split_mask_name:
                        adata.obsp[split] = adata.obsp.pop(split_mask_name)

                self.labels = labels

            else:
                raise ValueError("dataset_task_cfg.splits needs to be `all` or `predefined`")

        elif task_type == "binary":
            full_mask = np.sum([np.abs(adata.obsp[i]) for i in obsp_task_keys], axis=-1) > 0
            adata.obsp["full_mask"] = full_mask
            nz = np.nonzero(full_mask)

            num_tasks = len(obsp_task_keys)

            (labels := np.empty((len(nz[0]), num_tasks), dtype=np.float32)).fill(np.nan)
            for i, task in enumerate(obsp_task_keys):
                label_i = np.array(adata.obsp[task][nz]).ravel()
                labels[:, i][label_i == 1] = 1
                labels[:, i][label_i == -1] = 0
            self.labels = labels

        else:
            raise NotImplementedError(f"{task_type} Not Implemented Yet")

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
