import scanpy as sc
import anndata as ad
import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import hydra
import pandas as pd
from omegaconf import OmegaConf

from Heimdall.cell_representations import Cell_Representation



#####
# an example of some custom fg/fcs
#####
def identity_fg(adata_var):
    """
    identity_fg is an fg that returns a token id for each gene, effectively each gene
    is its own word.

    args:
        - adata_var: takes in the var dataframe, in this case, it expects the index to have the gene names

    output:
        - the output is a dictionary map between the gene names, and their corersponding token id for nn.embedding
    """
    print("> Performing the f_g identity, desc: each gene is its own token")
    gene_df = adata_var
    gene_mapping = {label: idx for idx, label in enumerate(gene_df.index.unique(), start=0)}
    return gene_mapping


def geneformer_fc(fg, adata):
    """
    geneformer_fc is a fc that will reprocess each cell by ordering them by their gene expression value,
    and replace each gene name by their corresponding representation, either token_id or a different vector

    right now this only supports token_id

    args:
        - fg: dictionary that maps gene names to token ids
        - adata: the whole, already processed, anndata object with the CellxGene Matrix

    output:
        - output: dataset, a numpy object that is dimension CellxGene where the position has the token denoting what gene it is
    """

    assert all(isinstance(value, (int)) for value in fg.values()), \
            "Current geneformer_fc only supports token ids"

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



### using @hydra.main so that we can take in command line arguments
@hydra.main(config_path="config", config_name="config", version_base = "1.3")
def main(config):
    print(OmegaConf.to_yaml(config))

    #####
    # For more details please check out the Cell_Representation object and the corresponding functions below
    #####
    
    CR = Cell_Representation(config) ## takes in the whole config from hydra
    CR.preprocess_anndata() ## standard sc preprocessing can be done here
    CR.preprocess_f_g(identity_fg) ## takes in the identity f_g specified above
    CR.preprocess_f_c(geneformer_fc) ## takes in the geneformer f_c specified above
    CR.prepare_labels() ## prepares the labels

    ## we can take this out here now and pass this into a PyTorch dataloader and separately create the model
    X = CR.cell_representation
    y = CR.labels

    print(f"Cell representation X: {X.shape}")
    print(f"Cell labels y: {y.shape}")

    return






if __name__ == '__main__':
    main()
