def identity_fg(adata_var):
    """Identify gene function.

    Returns a token id for each gene, effectively each gene is its own word.

    Args:
        adata_var: takes in the var dataframe, in this case, it expects the
            index to have the gene names.

    Return:
        A dictionary map between the gene names, and their corersponding token
        id for nn.embedding.

    """
    print("> Performing the f_g identity, desc: each gene is its own token")
    gene_df = adata_var
    gene_mapping = {label: idx for idx, label in enumerate(gene_df.index.unique(), start=0)}
    return gene_mapping
