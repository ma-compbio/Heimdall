import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_gene_medians(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        gene_medians = pickle.load(f)
    return gene_medians


def value_binning(expression_values, n_bins=10):
    """Bin the expression values into n_bins bins.

    Args:
        expression_values: array of expression values for a single cell.
        n_bins: number of bins.

    Return:
        Array of binned expression values.

    """
    if np.max(expression_values) == 0:
        return np.zeros_like(expression_values, dtype=np.int64)
    non_zero_values = expression_values[expression_values > 0]

    bin_edges = np.quantile(non_zero_values, np.linspace(0, 1, n_bins - 1))

    binned_values = np.digitize(expression_values, bin_edges, right=True)
    return binned_values


def old_geneformer_fc(fg, adata):
    """geneformer_fc is a fc that will reprocess each cell by ordering them by
    their gene expression value, and replace each gene name by their
    corresponding representation, either token_id or a different vector.

    Reprocesses each cell by ordering them by their gene expression value, and
    replace each gene name by their corresponding representation, either
    ``token_id`` or a different vector.

    Note:
        Currently, this only supports ``token_id``.

    Args:
        fg: dictionary that maps gene names to token ids.
        adata: the whole, already processed, anndata object with the CellxGene
            matrix.

    Return:
        A numpy object that is dimension CellxGene where the position has the
        token denoting what gene it is.

    """

    assert all(isinstance(value, (int)) for value in fg.values()), "Current geneformer_fc only supports token ids"

    print("> Performing the f_c using rank-based values, as seen in geneformer")
    df = pd.DataFrame(adata.X, columns=fg.keys())
    gene_medians = df.median()
    normalized_df = df.apply(lambda x: x / gene_medians[x.name])

    dataset = []
    for i in tqdm(range(len(normalized_df))):
        cell = normalized_df.iloc[i]
        sorted_cell = cell.sort_values(ascending=False).index
        cell_w_gene_ids = [fg[gene] for gene in sorted_cell]
        dataset.append(cell_w_gene_ids)

    dataset = np.array(dataset)
    return dataset


def geneformer_fc(fg, adata, embedding_layer=None):
    """geneformer_fc is a fc that will reprocess each cell by ordering them by
    their normalized gene expression value, and replace each gene name by their
    corresponding representation, either token_id or a different vector.

    right now this only supports token_id

    args:
        - fg: dictionary that maps gene names to token ids
        - adata: the whole, already processed, anndata object with the CellxGene Matrix

    output:
        - output: dataset, a numpy object that is dimension CellxGene where the position has the token denoting what
          gene it is

    """
    assert all(isinstance(value, (int)) for value in fg.values()), "Current geneformer_fc only supports token ids"

    print("> Performing the f_c using rank-based values, as seen in geneformer")

    # normalize by gene medians
    df = (
        pd.DataFrame(adata.X.toarray(), columns=fg.keys())
        if hasattr(adata.X, "toarray")
        else pd.DataFrame(adata.X, columns=fg.keys())
    )
    gene_medians = df.median()
    normalized_df = df.apply(lambda x: x / gene_medians[x.name])

    dataset = []
    for i in tqdm(range(len(normalized_df))):
        cell = normalized_df.iloc[i]
        sorted_cell = cell.sort_values(ascending=False).index
        # Use token ids only
        cell_w_gene_ids = [fg[gene] for gene in sorted_cell]

        dataset.append(cell_w_gene_ids)

    dataset = np.array(dataset)
    return dataset


def scgpt_fc(fg, adata, embedding_layer=None, num_bins=10):
    """scgpt_fc reprocesses each cell by binning genes based on expression
    values and replacing each gene name with their corresponding token_id.

    args:
        - fg: dictionary that maps gene names to token ids
        - adata: the whole, already processed, anndata object with the CellxGene Matrix
        - num_bins: number of bins for value binning

    output:
        - dataset: a numpy object that is dimension CellxGene where the position has the token denoting what gene it is
        - binned_values_dataset: a numpy object of binned expression values

    """
    # assert all(isinstance(value, int) for value in fg.values()), \
    #         "Current scgpt_fc only supports token ids"

    print("> Performing the f_c using rank-based values with binning, as seen in scGPT")
    df = (
        pd.DataFrame(adata.X.toarray(), columns=fg.keys())
        if hasattr(adata.X, "toarray")
        else pd.DataFrame(adata.X, columns=fg.keys())
    )
    df = df[df.columns.intersection(fg.keys())]
    binned_values_dataset = []

    for i in tqdm(range(len(df))):
        cell = df.iloc[i]
        # apply quantile-based binning to the expression values
        binned_values = value_binning(cell.values, n_bins=num_bins)

        binned_values_dataset.append(binned_values)

    binned_values_dataset = np.array(binned_values_dataset)
    return binned_values_dataset
