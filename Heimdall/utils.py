import importlib
import math
import warnings
from functools import partial, wraps
from pprint import pformat
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import default_collate

MAIN_KEYS = {"inputs", "labels", "masks"}


def instantiate_from_config(
    config: DictConfig,
    *args: Tuple[Any],
    _target_key: str = "type",
    _params_key: str = "args",
    _disable_key: str = "disable",
    _catch_conflict: bool = True,
    **extra_kwargs: Any,
):
    if config.get(_disable_key, False):
        return

    # Obtain target object and kwargs
    module, obj = config[_target_key].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), obj)
    kwargs = config.get(_params_key, None) or {}

    if _catch_conflict:
        assert not (set(kwargs) & set(extra_kwargs)), f"kwargs and extra_kwargs conflicted:\n{kwargs=}\n{extra_kwargs=}"
    full_kwargs = {**kwargs, **extra_kwargs}

    # Instantiate object and handel exception during instantiation
    try:
        return cls(*args, **full_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {cls!r} with\nargs:\n{pformat(args)}\nkwargs:\n{pformat(full_kwargs)}",
        ) from e


def get_value(dictionary, key, default=False):
    return dictionary.get(key, default)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):  # , dropout: float = 0.1
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.einsum("sbe->bse", pe)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Forward function.

        Args:
            x: Tensor, shape ``[batch_size , seq_len, embedding_dim]``

        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:, : x.size(1)]  # Broadcasting to match input shape
        x = self.dropout(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Dataset Preparation collation tool
def heimdall_collate_fn(examples):
    """Heimdall data collate function.

    This function helps the dataloader prepare the dataset into a consistent
    format, specifically the dataset is likely prepared as such:

    .. code-block:: python

        ds_train = Dataset.from_dict({"inputs": train_x,
                                      'labels':train_y,
                                      'conditional_tokens_1': train_x,
                                      'conditional_tokens_2': train_x})

    where the  `conditional_tokens_*` are optional conditional tokens. This
    will process the output of a batch to be a dictionary with keys: "inputs",
    "labels" (these are mandatory), and "conditional_tokens" which is a
    dictionary of the conditional tokens.

    """
    # batch = {}
    # # Assume all examples have the same keys, use the keys from the first example
    # keys = examples[0].keys()
    # conditional_tokens = {}

    # for key in keys:
    #     if key in ["inputs", "labels"]:
    #         # Check if the data needs to be stacked or just converted to tensor
    #         if isinstance(examples[0][key], list):  # or any other condition to decide on stacking
    #             # Stack tensors if the data type is appropriate (e.g., lists of numbers)
    #             batch[key] = torch.stack([torch.tensor(example[key]) for example in examples])
    #         else:
    #             # Convert to tensor directly if it's a singular item like labels
    #             batch[key] = torch.tensor([example[key] for example in examples])

    #     else:  # if it is not an input or label, it is automatically processed as a conditional token
    #         if isinstance(examples[0][key], list):
    #             conditional_tokens[key] = torch.stack([torch.tensor(example[key]) for example in examples])
    #         else:
    #             conditional_tokens[key] = torch.tensor([example[key] for example in examples])
    # batch["conditional_tokens"] = conditional_tokens
    # return batch

    # Collate batch using pytorch's default collate function
    flat_batch = default_collate(examples)

    # Regroup by keys
    batch, conditional_tokens = {}, {}
    for key, val in flat_batch.items():
        (batch if key in MAIN_KEYS else conditional_tokens)[key] = val

    if conditional_tokens:
        batch["conditional_tokens"] = conditional_tokens

    return batch


def deprecate(func: Optional[Callable] = None, raise_error: bool = False):

    if func is None:
        return partial(deprecate, raise_error=raise_error)

    @wraps(func)
    def bounded(*args, **kwargs):
        msg = f"{func} is deprecated, do not use"
        if raise_error:
            raise RuntimeError(msg)

        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return bounded
