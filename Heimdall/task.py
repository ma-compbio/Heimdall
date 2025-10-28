import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

if TYPE_CHECKING:
    from Heimdall.cell_representations import CellRepresentation

from Heimdall.utils import instantiate_from_config

CellFeatType = NDArray[np.int_] | NDArray[np.float32]
FeatType = CellFeatType | tuple[CellFeatType, CellFeatType]
LabelType = NDArray[np.int_] | NDArray[np.float32]


@dataclass
class Task(ABC):
    """Heimdall task key-value store.

    Contains information about an scFM task and training details.

    """

    data: "CellRepresentation"
    task_type: str
    metrics: list[str]
    shuffle: bool
    batchsize: int
    epochs: int
    dataset_config: DictConfig
    head_config: DictConfig
    loss_config: DictConfig
    interaction_type: str | None = None
    top_k: list[int] | None = None
    label_obsm_name: str | None = None
    label_col_name: str | None = None
    reducer_config: DictConfig | None = None
    splits: DictConfig | None = None
    train_split: float | None = (None,)
    track_metric: str | None = None
    early_stopping: bool = False
    early_stopping_patience: int = 5

    @property
    def labels(self) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        return getattr(self, "_labels", None)

    @labels.setter
    def labels(self, val) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        self._labels = val

    @property
    def num_tasks(self) -> int:
        if "_num_tasks" not in self.__dict__:
            warnings.warn(
                "Need to improve to explicitly handle multiclass vs. multilabel",
                UserWarning,
                stacklevel=2,
            )
            assert self.task_type in [
                "regression",
                "binary",
                "multiclass",
                "mlm",
            ], "task type must be regression, binary, multiclass or mlm. Check the task config file."

            task_type = self.task_type
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
            elif task_type == "mlm":
                # out = self._labels.max() + 1
                out = self.labels.shape[0] + 1  # TODO why +1 ?
            else:
                raise ValueError(
                    f"Unknown task type {task_type!r}. Valid options are: 'multiclass', 'binary', 'regression', 'mlm'.",
                )

            self._num_tasks = out = int(out)
            print(
                f"> Task dimension: {out} " f"(task type {self.task_type!r}, {self.labels.shape=})",
            )

        return self._num_tasks

    @property
    def idx(self) -> NDArray[np.int_]:
        return self.data._idx

    @abstractmethod
    def setup_labels(self): ...

    def get_inputs(self, idx, shared_inputs):
        return {
            "labels": self.labels[idx],
        }


class SingleInstanceTask(Task):
    def setup_labels(self):
        adata = self.data.adata
        if self.label_col_name is not None:
            assert self.label_obsm_name is None
            df = adata.obs
            class_mapping = {
                label: idx
                for idx, label in enumerate(
                    df[self.label_col_name].unique(),
                    start=0,
                )
            }
            df["class_id"] = df[self.label_col_name].map(class_mapping)
            labels = np.array(df["class_id"])
            if self.task_type == "regression":
                labels = labels.reshape(-1, 1).astype(np.float32)

        elif self.label_obsm_name is not None:
            assert self.label_col_name is None
            df = adata.obsm[self.label_obsm_name]

            if self.task_type == "binary":
                (labels := np.empty(df.shape, dtype=np.float32)).fill(np.nan)
                labels[np.where(df == 1)] = 1
                labels[np.where(df == -1)] = 0
            elif self.task_type == "regression":
                labels = np.array(df).astype(np.float32)

            print(f"labels shape {labels.shape}")

        else:
            raise ValueError("Either 'label_col_name' or 'label_obsm_name' needs to be set.")

        self.labels = labels


class PairedInstanceTask(Task):
    def setup_labels(self):
        adata = self.data.adata
        full_mask = adata.obsp["full_mask"]
        nz = np.nonzero(full_mask)

        # Task type specific handling
        task_type = self.task_type
        if task_type == "multiclass":
            if len(self.data.obsp_task_keys) > 1:
                raise ValueError(
                    f"{task_type!r} only supports a single task key, provided task keys: {self.data.obsp_task_keys}",
                )

            task_mat = adata.obsp[self.data.obsp_task_keys[0]]
            num_tasks = task_mat.max()  # class id starts from 1. 0's are ignoreed
            labels = np.array(task_mat[nz]).ravel().astype(np.int64) - 1  # class 0 is not used

        elif task_type == "binary":
            num_tasks = len(self.data.obsp_task_keys)

            (labels := np.empty((len(nz[0]), num_tasks), dtype=np.float32)).fill(np.nan)
            for i, task in enumerate(self.data.obsp_task_keys):
                label_i = np.array(adata.obsp[task][nz]).ravel()
                labels[:, i][label_i == 1] = 1
                labels[:, i][label_i == -1] = 0

        elif task_type == "regression":
            num_tasks = len(self.data.obsp_task_keys)

            labels = np.zeros((len(nz[0]), num_tasks), dtype=np.float32)
            for i, task in enumerate(self.data.obsp_task_keys):
                labels[:, i] = np.array(adata.obsp[task][nz]).ravel()

        else:
            raise ValueError(f"task_type must be one of: 'multiclass', 'binary', 'regression'. Got: {task_type!r}")

        self.labels = labels


class MLMMixin:
    def get_inputs(self, idx, shared_inputs):
        identity_inputs = shared_inputs["identity_inputs"]
        return {
            "identity_inputs": identity_inputs,
            "labels": identity_inputs.astype(int),
        }

    def setup_labels(self):
        # Dummy labels to indicate task size
        self.labels = np.empty(self.data.fg.vocab_size)


class MaskedMixin(ABC):
    def __init__(self, *args, mask_ratio: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio

    @property
    @abstractmethod
    def mask_token(self): ...


class TransformationMixin(ABC):
    def get_inputs(self, idx, shared_inputs):
        data = super().get_inputs(idx, shared_inputs)
        return self._transform(data)

    @abstractmethod
    def _transform(self, data): ...


class SeqMaskedMLMTask(TransformationMixin, MaskedMixin, MLMMixin, SingleInstanceTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._num_tasks = self.data.adata.n_vars  # number of genes

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


class Tasklist:
    """Container for multiple Heimdall tasks.

    Tasks must use the same `Dataset` object config and splits/dataloader.

    """

    PROPERTIES = (
        "splits",
        "dataset_config",
        "shuffle",
        "batchsize",
        "epochs",
        "interaction_type",
        "early_stopping",
        "early_stopping_patience",
    )

    def __init__(
        self,
        data: "CellRepresentation",
        subtask_configs: DictConfig | dict,
    ):

        self.data = data
        self._tasks = {
            subtask_name: instantiate_from_config(subtask_config, data)
            for subtask_name, subtask_config in subtask_configs.items()
        }

        self.set_unique_properties()
        self.num_subtasks = len(self._tasks)

    def set_unique_properties(self):
        for property_name in self.PROPERTIES:
            unique_properties = {getattr(task, property_name, None) for task in self._tasks.values()}
            if len(unique_properties) > 1:
                raise ValueError(f"All tasks must use the same `{property_name}` value.")

            unique_property = next(iter(unique_properties))
            setattr(self, property_name, unique_property)

    def __getitem__(self, key: str | None):
        if key is None:
            if len(self._tasks) > 1:
                raise ValueError("`None` key only works if `TaskList` contains a singular item.")

            return next(iter(self._tasks.values()))

        return self._tasks[key]

    def __setitem__(self, key: str, value: Task):
        self._tasks[key] = value
        self.num_subtasks = len(self._tasks)

    def __delitem__(self, key: str):
        del self._tasks[key]
        self.num_subtasks = len(self._tasks)

    def __iter__(self):
        yield from self._tasks.items()

    def __len__(self):
        return self.num_subtasks
