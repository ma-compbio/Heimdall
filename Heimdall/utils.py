import hashlib
import importlib
import json
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial, wraps
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import awkward as ak
import mygene
import numpy as np
import pandas as pd
import requests
from numpy.random import Generator
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import default_collate
from tqdm.auto import tqdm

MAIN_KEYS = {
    "identity_inputs",
    "expression_inputs",
    "labels",
    "masks",
    "expression_padding",
    "adjacency_matrix",
    "subgraph_indices",
}


def hash_config(cfg: DictConfig) -> str:
    """Generate hash for a given config."""
    cfg_str = OmegaConf.to_yaml(cfg, sort_keys=True)
    hex_str = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex_str))


def get_cached_paths(cfg: DictConfig, cache_dir: Path, file_name: str) -> Tuple[Path, Path]:
    """Get cached data and config path given config."""
    hash_str = hash_config(cfg)

    cache_dir = cache_dir / hash_str
    cache_dir.mkdir(exist_ok=True, parents=True)

    cached_file_path = cache_dir / file_name
    cached_cfg_path = cache_dir / "config.yaml"

    return cached_file_path, cached_cfg_path


def searchsorted2d(bin_edges: NDArray, expression: NDArray, side: str = "left"):
    """Vectorization of `np.searchsorted` for 2D `bin_edges` array.

    Adds offset to each row of `bin_edges` and `expression` to make sure that rows of the two inputs correspond
    uniquely to each other. This trades off algorithmic efficiency for vectorization.

    See https://stackoverflow.com/a/40588862/13952002

    Args:
        bin_edges: per-cell bin_edges
        expression: raw expression as integer count

    """

    num_cells = len(expression)
    max_value = np.maximum(ak.ptp(bin_edges), ak.ptp(expression)) + 1

    cell_indices = np.arange(num_cells)[:, np.newaxis]
    offsets = ak.Array(max_value * cell_indices)

    expression_counts = ak.count(expression, axis=1)
    bin_edges_counts = ak.count(bin_edges, axis=1)
    offset_bin_edges = ak.ravel(bin_edges + offsets).to_numpy()
    offset_expression = ak.ravel(expression + offsets).to_numpy()
    binned_values = np.searchsorted(offset_bin_edges, offset_expression, side=side)

    binned_values = ak.unflatten(binned_values, expression_counts)
    cumulative_counts = np.cumsum([0] + ak.to_list(bin_edges_counts))[:-1]

    binned_values = binned_values - cumulative_counts

    return binned_values


def get_name(target: str):
    module, obj = target.rsplit(".", 1)
    name = getattr(importlib.import_module(module, package=None), obj)

    return name, module, obj


def instantiate_from_config(
    config: DictConfig,
    *args: Tuple[Any],
    _target_key: str = "type",
    _constructor_key: str = "constructor",
    _params_key: str = "args",
    _disable_key: str = "disable",
    _catch_conflict: bool = True,
    return_name: bool = False,
    **extra_kwargs: Any,
):
    if config.get(_disable_key, False):
        return

    # Obtain target object and kwargs
    cls, module, obj = get_name(config[_target_key])
    kwargs = config.get(_params_key, None) or {}

    constructor_name = config.get(_constructor_key, None)
    if constructor_name is not None:
        constructor = getattr(cls, constructor_name)
    else:
        constructor = cls

    if _catch_conflict:
        assert not (set(kwargs) & set(extra_kwargs)), f"kwargs and extra_kwargs conflicted:\n{kwargs=}\n{extra_kwargs=}"
    full_kwargs = {**kwargs, **extra_kwargs}

    # Instantiate object and handel exception during instantiation
    try:
        if return_name:
            return constructor(*args, **full_kwargs), obj

        return constructor(*args, **full_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {constructor!r} with\nargs:\n{pformat(args)}\nkwargs:\n{pformat(full_kwargs)}",
        ) from e


def get_value(dictionary, key, default=False):
    return dictionary.get(key, default)


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


MG = mygene.MyGeneInfo()

ENSEMBL_URL_MAP = {
    "human": "https://ftp.ensembl.org/pub/release-{}/gtf/homo_sapiens/Homo_sapiens.GRCh38.{}.gtf.gz",
    "mouse": "https://ftp.ensembl.org/pub/release-{}/gtf/mus_musculus/Mus_musculus.GRCm39.{}.gtf.gz",
}


@dataclass
class GeneMappingOutput:
    """Gene mapping results data structure.

    Attributes:
        genes: List of original gene ids.
        mapping_full: Dictionary mapping from query gene id to target ids as
            a list. If the mapping is not available, then this gene id would
            not appear in this mapping. Thus, this dictionary mmight contain
            less elements than the total number of the queried genes.
        mapping_combined: Dictionary mapping from query gene id to combined
            target ids (concatenated by "|", e.g., ["id1", "id2"] would be
            converted to "id1|id2"). If the gene id conversion is not
            applicable, then we will map it to "N/A". Thus, this dictionary
            contains the same number of element as the total number of the
            queried genes.
        mapping_reduced: Similar to mapping_combined, but only use the first
            mapped ids (sorted alphabetically) when multiple ids are available.
            Furthermore, use the query gene id as the target gene id if the
            mapping is unavailable. This dictionary contains the same number of
            elements as the total number of the queried genes.

    """

    genes: List[str]
    mapping_full: Dict[str, List[str]]
    mapping_combined: Dict[str, str] = field(init=False)
    mapping_reduced: Dict[str, str] = field(init=False)

    def __post_init__(self):
        self.mapping_combined = {}
        self.mapping_reduced = {}
        for g in self.genes:
            if (ensembl := self.mapping_full.get(g)) is None:
                self.mapping_combined[g] = "N/A"
                self.mapping_reduced[g] = g
            else:
                ensembl = sorted(set(ensembl))
                self.mapping_full[g] = ensembl
                self.mapping_reduced[g] = ensembl[0]
                self.mapping_combined[g] = "|".join(ensembl)

        print(
            f"Successfully mapped {len(self.mapping_full):,} out of {len(self.genes):,} "
            f"genes ({len(self.mapping_full) / len(self.genes):.1%})",
        )


def symbol_to_ensembl(
    genes: List[str],
    species: str = "human",
    extra_query_kwargs: Optional[Dict[str, Any]] = None,
) -> GeneMappingOutput:
    # Query from MyGene.Info server
    print(f"Querying {len(genes):,} genes")
    query_results = MG.querymany(
        genes,
        species=species,
        scopes="symbol",
        fields="ensembl.gene",
        **(extra_query_kwargs or {}),
    )

    # Unpack query results
    symbol_to_ensembl_dict = {}
    for res in query_results:
        symbol = res["query"]
        if (ensembl := res.get("ensembl")) is None:
            continue

        if isinstance(ensembl, dict):
            new_ensembl_genes = [ensembl["gene"]]
        elif isinstance(ensembl, list):
            new_ensembl_genes = [i["gene"] for i in ensembl]
        else:
            raise ValueError(f"Unknown ensembl query result type {type(ensembl)}: {ensembl!r}")

        symbol_to_ensembl_dict[symbol] = symbol_to_ensembl_dict.get(symbol, []) + new_ensembl_genes

    return GeneMappingOutput(genes, symbol_to_ensembl_dict)


def symbol_to_ensembl_from_ensembl(
    data_dir: Union[str, Path],
    genes: List[str],
    species: str = "human",
    release: int = 112,
) -> GeneMappingOutput:
    mapping_dict = _load_ensembl_table(data_dir, species, release)
    symbol_to_ensembl_dict = {i: mapping_dict[i] for i in genes if i in mapping_dict}
    return GeneMappingOutput(genes, symbol_to_ensembl_dict)


def _load_ensembl_table(
    data_dir: Union[str, Path],
    species: str,
    release: int,
) -> Dict[str, List[str]]:
    try:
        url = ENSEMBL_URL_MAP[species].format(release, release)
        fname = url.split("/")[-1]
    except KeyError as e:
        raise KeyError(
            f"Unknown species {species!r}, available options are {sorted(ENSEMBL_URL_MAP)}",
        ) from e

    data_dir = Path(data_dir).resolve() / "gene_mapping" / "ensembl" / species
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mapping data directory: {data_dir}")

    raw_path = data_dir / fname
    attr_path = data_dir / "gene_attr_table.tsv.gz"
    mapping_symbol_to_ensembl_path = data_dir / "symbol_to_ensembl.json"
    mapping_ensembl_to_symbol_path = data_dir / "ensembl_to_symbol.json"

    # Download GTF from Ensembl
    if not raw_path.is_file():
        print(f"Downloading gene annotation from {url}")
        with requests.get(url) as r:
            if not r.ok:
                raise requests.RequestException(f"Fail to download {url} ({r})")

            with open(raw_path, "wb") as f:
                f.write(r.content)

    # Prepare gene attribute table from GTF
    if not attr_path.is_file():
        print("Extracting gene attributes")
        full_df = pd.read_csv(raw_path, sep="\t", compression="gzip", comment="#", header=None, low_memory=False)

        def flat_to_dict(x: str) -> Tuple[str, str]:
            """Convert a flat string to a dictionary.

            Example:
                'gene_id "xxx"; gene_name "ABC";' -> {"gene_id": "xxx", "gene_name": "ABC"}

            """
            data = []
            for i in x.split("; "):
                key, val = i.split(" ", 1)
                data.append((key.strip('"'), val.rstrip(";").strip('"')))
            return dict(data)

        attr_df = pd.DataFrame(list(map(flat_to_dict, tqdm(full_df[8]))))
        attr_df.to_csv(attr_path, sep="\t", index=False, compression="gzip")

    # Prepare symbol-ensembl mappings
    if not mapping_symbol_to_ensembl_path.is_file():
        print("Preparing gene ID mapping")
        attr_df = pd.read_csv(attr_path, sep="\t", compression="gzip")
        mapping_df = attr_df[["gene_id", "gene_name"]].drop_duplicates()

        symbol_to_ensembl, ensembl_to_symbol = defaultdict(list), defaultdict(list)
        for _, (ensembl, symbol) in mapping_df.iterrows():
            symbol_to_ensembl[symbol].append(ensembl)
            ensembl_to_symbol[ensembl].append(symbol)

        symbol_to_ensembl, ensembl_to_symbol = map(  # noqa: C417
            lambda x: {i: sorted(j) for i, j in x.items()},
            (symbol_to_ensembl, ensembl_to_symbol),
        )

        with (
            open(mapping_symbol_to_ensembl_path, "w") as f1,
            open(mapping_ensembl_to_symbol_path, "w") as f2,
        ):
            json.dump(symbol_to_ensembl, f1, indent=4)
            json.dump(ensembl_to_symbol, f2, indent=4)
        print(f"Mapping saved to cache: {mapping_symbol_to_ensembl_path}")

    else:
        print(f"Loading mapping from cache: {mapping_symbol_to_ensembl_path}")
        with open(mapping_symbol_to_ensembl_path) as f:
            symbol_to_ensembl = json.load(f)

    return symbol_to_ensembl


def sample_without_replacement(
    rng: Generator,
    max_index: int,
    num_samples: int,
    sample_size: int,
    attention_mask: Tensor,
):
    """Generate random samples of indices without replacement using NumPy
    vectorization.

    Args:
        rng: random number generator from which to sample
        max_index: max index of which can be included in a sample
        num_samples: number of index vectors to sample
        sample_size: number of random indices (without replacement) per sample

    Return:
        randomly sampled indices without replacement

    """
    attention_mask = np.array(attention_mask)
    random_samples = rng.random((num_samples, max_index))
    random_indices = np.argpartition(random_samples, sample_size - 1, axis=1)[:, :sample_size]

    return random_indices


def get_dtype(dtype_name: str, backend: str = "torch"):
    """Retrieve `dtype` object from backend library."""

    if backend == "torch" and dtype_name == "float16":
        dtype_name = "bfloat16"  # Promote float16 dtype for Torch backend

    dtype, module_name, dtype_name = get_name(f"{backend}.{dtype_name}")

    return dtype
