import numpy as np
import pandas as pd
from tqdm import tqdm


def geneformer_fc(fg, adata):
    """geneformer_fc is a fc that will reprocess each cell by ordering them by
    their gene expression value, and replace each gene name by their
    corresponding representation, either token_id or a different vector.

    right now this only supports token_id

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

    dataset = []
    for i in tqdm(range(len(df))):
        cell = df.iloc[i]
        sorted_cell = cell.sort_values(ascending=False).index
        cell_w_gene_ids = [fg[gene] for gene in sorted_cell]
        dataset.append(cell_w_gene_ids)

    dataset = np.array(dataset)
    return dataset
