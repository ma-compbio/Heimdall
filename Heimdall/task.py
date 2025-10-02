import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from Heimdall.cell_representations import CellRepresentation


@dataclass
class Task:
    """Heimdall task key-value store.

    Contains information about an scFM task and training details.

    """

    data: CellRepresentation
    task_type: str
    metrics: list[str]
    shuffle: bool
    batchsize: int
    epochs: int
    dataset_config: DictConfig
    head_config: DictConfig
    cell_rep_config: DictConfig
    loss_config: DictConfig
    interaction_type: str | None = None
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
        return self._labels

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
                out = self._labels.shape[0] + 1  # TODO why +1 ?
            else:
                raise ValueError(
                    f"Unknown task type {task_type!r}. Valid options are: 'multiclass', 'binary', 'regression', 'mlm'.",
                )

            self._num_tasks = out = int(out)
            print(
                f"> Task dimension: {out} " f"(task type {self.task_type!r}, {self.labels.shape=})",
            )

        return self._num_tasks
