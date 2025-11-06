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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import anndata as ad
import awkward as ak
import mygene
import numpy as np
import pandas as pd
import requests
import scanpy as sc
from anndata.abc import CSCDataset, CSRDataset
from matplotlib import pyplot as plt
from numpy.random import Generator
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from scipy import sparse as sp
from torch import Tensor
from torch.utils.data import default_collate
from tqdm.auto import tqdm

import wandb

if TYPE_CHECKING:
    from Heimdall.cell_representations import CellRepresentation

INPUT_KEYS = {
    "identity_inputs",
    "expression_inputs",
    "masks",
    "expression_padding",
}

MAIN_KEYS = {
    *INPUT_KEYS,
    "labels",
    # "adjacency_matrix",
    # "subgraph_indices",
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
def get_collation_closure(keys=MAIN_KEYS):
    """Heimdall data collate function."""

    def collate_fn(batch):
        collated = {}
        first_sample = batch[0]
        for key in keys:
            inner_dict = {}
            for subtask_name in first_sample[key]:
                values = [b[key][subtask_name] for b in batch]
                # Drop Nones, or replace with zeros
                is_invalid = [v is None for v in values]
                if all(is_invalid):
                    inner_dict[subtask_name] = None
                elif any(is_invalid):
                    raise ValueError("Cannot have multiple samples with inhomogenous input validities.")
                else:
                    inner_dict[subtask_name] = default_collate(values)
            collated[key] = inner_dict
        return dict(collated)

    return collate_fn


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
            not appear in this mapping. Thus, this dictionary might contain
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

    identifiers: List[str]
    mapping_full: Dict[str, List[str]]
    verbose: int | bool = False
    mapping_combined: Dict[str, str] = field(init=False)
    mapping_reduced: Dict[str, str] = field(init=False)

    def __post_init__(self):
        self.mapping_combined = {}
        self.mapping_reduced = {}
        for identifier in self.identifiers:
            hits = self.mapping_full.get(identifier)
            if hits is None or pd.isna(hits[0]):
                self.mapping_combined[identifier] = "N/A"
                self.mapping_reduced[identifier] = identifier
            else:
                hits = sorted(set(hits))
                self.mapping_full[identifier] = hits
                self.mapping_reduced[identifier] = hits[0]
                self.mapping_combined[identifier] = "|".join(hits)

        map_ratio = len(self.mapping_full) / len(self.identifiers)
        conditional_print(
            (
                f"Successfully mapped {len(self.mapping_full):,} out of {len(self.identifiers):,} "
                f"identifiers ({map_ratio:.1%})"
            ),
            condition=self.verbose,
        )


def symbol_to_ensembl(
    genes: List[str],
    species: str = "human",
    extra_query_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int | bool = False,
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

    return GeneMappingOutput(genes, symbol_to_ensembl_dict, verbose)


def symbol_to_ensembl_from_ensembl(
    data_dir: Union[str, Path],
    symbols: List[str] = None,
    ensembl_ids: List[str] = None,
    species: str = "human",
    release: int = 112,
    verbose: int | bool = False,
) -> GeneMappingOutput:

    symbol_to_ensembl, ensembl_to_symbol = _load_ensembl_table(data_dir, species, release, verbose=verbose)

    if symbols is not None:
        symbol_to_ensembl_dict = {i: symbol_to_ensembl[i] for i in symbols if i in symbol_to_ensembl}
        mapping = GeneMappingOutput(symbols, symbol_to_ensembl_dict, verbose)
    elif ensembl_ids is not None:
        ensembl_to_symbol_dict = {i: ensembl_to_symbol[i] for i in ensembl_ids if i in ensembl_to_symbol}
        mapping = GeneMappingOutput(ensembl_ids, ensembl_to_symbol_dict, verbose)

    return mapping


def _prepare_gene_attr_table(
    raw_path: Union[str, Path],
    attr_path: Union[str, Path],
):
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


def _prepare_symbol_ensembl_maps(
    attr_path: Union[str, Path],
    mapping_symbol_to_ensembl_path: Union[str, Path],
    mapping_ensembl_to_symbol_path: Union[str, Path],
):
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

    return symbol_to_ensembl, ensembl_to_symbol


def _load_ensembl_table(
    data_dir: Union[str, Path],
    species: str,
    release: int,
    verbose: int | bool = False,
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
    conditional_print(f"Mapping data directory: {data_dir}", verbose)

    raw_path = data_dir / fname
    attr_path = data_dir / "gene_attr_table.tsv.gz"
    mapping_symbol_to_ensembl_path = data_dir / "symbol_to_ensembl.json"
    mapping_ensembl_to_symbol_path = data_dir / "ensembl_to_symbol.json"

    # Download GTF from Ensembl
    if not raw_path.is_file():
        conditional_print(f"Downloading gene annotation from {url}", verbose)
        with requests.get(url) as r:
            if not r.ok:
                raise requests.RequestException(f"Fail to download {url} ({r})")

            with open(raw_path, "wb") as f:
                f.write(r.content)

    # Prepare gene attribute table from GTF
    if not attr_path.is_file():
        conditional_print("Extracting gene attributes", verbose)
        _prepare_gene_attr_table(raw_path, attr_path)

    # Prepare symbol-ensembl mappings
    if not mapping_symbol_to_ensembl_path.is_file():
        symbol_to_ensembl, ensembl_to_symbol = _prepare_symbol_ensembl_maps(
            attr_path,
            mapping_symbol_to_ensembl_path,
            mapping_ensembl_to_symbol_path,
        )
    else:
        conditional_print(f"Loading mapping from cache: {mapping_symbol_to_ensembl_path}", verbose)
        with open(mapping_symbol_to_ensembl_path) as f:
            symbol_to_ensembl = json.load(f)

        with open(mapping_ensembl_to_symbol_path) as f:
            ensembl_to_symbol = json.load(f)

    return symbol_to_ensembl, ensembl_to_symbol


def convert_to_ensembl_ids(adata, data_dir, species="human", verbose: int | bool = False):
    """Converts gene symbols in the anndata object to Ensembl IDs using a
    provided mapping.

    Args:
        - adata: anndata object with gene symbols as var index
        - data_dir: directory where the data is stored
        - species: species name (default is "human")

    Returns:
        - data: anndata object with Ensembl IDs as var index
        - symbol_to_ensembl_mapping: mapping dictionary from symbols to Ensembl IDs

    """
    if (adata.var.index.str.startswith("ENS").sum() / len(adata.var.index)) < 0.9:
        gene_mapping = symbol_to_ensembl_from_ensembl(
            data_dir=data_dir,
            symbols=adata.var.index.tolist(),
            species=species,
            verbose=verbose,
        )
        adata.uns["gene_mapping:symbol_to_ensembl"] = gene_mapping.mapping_full

        adata.var["gene_symbol"] = adata.var.index
        adata.var["gene_ensembl"] = adata.var["gene_symbol"].map(
            gene_mapping.mapping_combined.get,
        )
        adata.var.index = adata.var.index.map(gene_mapping.mapping_reduced)
        adata.var.index.name = "index"
    else:
        gene_mapping = symbol_to_ensembl_from_ensembl(
            data_dir=data_dir,
            ensembl_ids=adata.var.index.tolist(),
            species=species,
            verbose=verbose,
        )

        adata.var["gene_ensembl"] = adata.var.index
        adata.var["gene_symbol"] = adata.var["gene_ensembl"].map(
            gene_mapping.mapping_combined.get,
        )

        adata.uns["gene_mapping:ensembl_to_symbol"] = gene_mapping.mapping_full

        # adata.var.index = adata.var.index.map(symbol_to_ensembl_mapping.mapping_reduced)
        adata.var.index.name = "index"

    return adata, gene_mapping


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


def _get_inputs_from_csr(data: "CellRepresentation", cell_index: int, drop_zeros: bool):
    """Get expression values and gene indices from internal CSR representation.

    Args:
        cell_index: cell for which to process expression values and get indices, as stored in `adata`.

    """

    adata = data.adata
    identity_valid_mask = data.fg.identity_valid_mask
    if drop_zeros is True:
        if issparse(adata.X):
            cell = adata.X[[cell_index], :].toarray().flatten()[identity_valid_mask]
            (cell_identity_inputs,) = cell.nonzero()
            cell_expression_inputs = cell[cell_identity_inputs]
        else:
            cell_expression_inputs_full = adata.X[cell_index, :][identity_valid_mask]
            (cell_identity_inputs,) = np.nonzero(cell_expression_inputs_full)
            cell_expression_inputs = cell_expression_inputs_full[cell_identity_inputs]
    else:
        cell_expression_inputs = adata.X[[cell_index], :].toarray().flatten()[identity_valid_mask]
        cell_identity_inputs = np.arange(data.num_genes)

    return cell_identity_inputs, cell_expression_inputs


def issparse(x):
    return sp.issparse(x) or isinstance(x, (CSRDataset, CSCDataset))


def save_umap(
    cr: "CellRepresentation",
    embeddings,
    savepath,
    embedding_name="heimdall_latents",
    split="test",
    log_umap: bool = False,
):
    def save_partition_umap(adata, embeddings, savepath, embedding_name):
        fig, ax = plt.subplots(1, figsize=(4, 4))
        adata.obsm[embedding_name] = embeddings

        sc.pp.neighbors(adata, use_rep=embedding_name)
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

        sc.pl.umap(adata, ax=ax, show=False)
        ad.io.write_h5ad(savepath, adata)

        return fig

    if hasattr(cr, "splits"):
        full_dataset = cr.datasets["full"]
        if hasattr(full_dataset, "partition_splits"):
            cumulative_sizes = np.cumsum(
                [
                    len(full_dataset.partition_splits[partition][split])
                    for partition in range(full_dataset.num_partitions)
                ],
            )
            cumulative_sizes = np.concatenate([[0], cumulative_sizes])
            for partition in range(full_dataset.num_partitions):
                start, end = cumulative_sizes[partition : partition + 2]
                full_dataset.partition = partition
                partition_savepath = Path(savepath)
                partition_savepath = partition_savepath.parent / f"partition_{partition}_{partition_savepath.name}"
                adata = cr.adata[cr.splits[split]].copy(savepath)
                fig = save_partition_umap(
                    adata=adata,
                    embeddings=embeddings[start:end],
                    savepath=partition_savepath,
                    embedding_name=embedding_name,
                )
                if log_umap:
                    wandb.log({f"{partition=}_{split}_umap": wandb.Image(fig)})
        else:
            adata = cr.adata[cr.splits[split]].copy()
            fig = save_partition_umap(
                adata=adata,
                embeddings=embeddings,
                savepath=savepath,
                embedding_name=embedding_name,
            )
            if log_umap:
                wandb.log({f"{split}_umap": wandb.Image(fig)})
    else:
        raise ValueError("No split information found.")


def conditional_print(msg: str, condition: bool):
    if condition:
        print(msg)


class AllPartitionsExhausted(Exception):
    def __init__(self, message: str = "All partitions exhausted"):
        super().__init__(message)


def project2simplex_(y, dim: int = 0, zero_threshold: float = 1e-10) -> Tensor:
    """(In-place) Projects a matrix such that the columns (or rows) lie on the
    unit simplex.

    See https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
    for a reference.

    The goal is to find a scalar mu such that || (y-mu)_+ ||_1 = 1

    Currently uses Newton's method to optimize || y - mu ||^2

    TODO: try implementing it this way instead: https://arxiv.org/pdf/1101.6081.pdf

    Args:
        y: list of vectors to be projected to unit simplex
        dim: dimension along which to project
        zero_threshold: threshold to treat as zero for numerical stability purposes

    """
    num_components = y.shape[dim]

    y.sub_(y.sum(dim=dim, keepdim=True).sub_(1), alpha=1 / num_components)
    mu = y.max(dim=dim, keepdim=True)[0].div_(2)
    derivative_prev, derivative = None, None
    for _ in range(num_components):
        difference = y.sub(mu)
        objective_value = difference.clip_(min=zero_threshold).sum(dim, keepdim=True).sub_(1)
        derivative = difference.gt_(zero_threshold).sum(dim, keepdim=True)

        if derivative_prev is not None and (derivative == derivative_prev).all():
            break

        mu.addcdiv_(objective_value, derivative)
        derivative_prev = derivative

    y.sub_(mu).clip_(min=zero_threshold)
    assert y.sum(dim=dim).sub_(1).abs_().max() < 1e-4, y.sum(dim=dim).sub_(1).abs_().max()
    return y
