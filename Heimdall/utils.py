import math

import torch
import torch.nn as nn


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
    batch = {}
    # Assume all examples have the same keys, use the keys from the first example
    keys = examples[0].keys()
    conditional_tokens = {}

    for key in keys:
        if key in ["inputs", "labels"]:
            # Check if the data needs to be stacked or just converted to tensor
            if isinstance(examples[0][key], list):  # or any other condition to decide on stacking
                # Stack tensors if the data type is appropriate (e.g., lists of numbers)
                batch[key] = torch.stack([torch.tensor(example[key]) for example in examples])
            else:
                # Convert to tensor directly if it's a singular item like labels
                batch[key] = torch.tensor([example[key] for example in examples])

        else:  # if it is not an input or label, it is automatically processed as a conditional token
            if isinstance(examples[0][key], list):
                conditional_tokens[key] = torch.stack([torch.tensor(example[key]) for example in examples])
            else:
                conditional_tokens[key] = torch.tensor([example[key] for example in examples])
    batch["conditional_tokens"] = conditional_tokens
    return batch
