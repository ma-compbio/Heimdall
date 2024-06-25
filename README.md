# Heimdall




# Installation


```
git clone 

cd Heimdall

conda create --name heimdall_empty python=3.10
conda activate heimdall
pip install -r requirements.txt

```


# Model Documentation

The `Heimdall_Transformer` object is a default transformer that is flexible for `learned` embeddings and `predefined` embeddings, as well as conditional tokens that can be `learned` or `predefined` as well. Here is an example:


```
## initialize the model
from Heimdall.models import Heimdall_Transformer, TransformerConfig

### canonical example of conditional tokens being added, learened
config = TransformerConfig(vocab_size = 1000, max_seq_length = 1000)

conditional_tokens = {
    "conditional_tokens_1":{
        "type": "learned", ## This can be learned or predefined
        "vocab_size": 1000
    }
}
model = Heimdall_Transformer(config=config, input_type="learned", conditional_input_types=conditional_tokens)

## Demo purposes
x = torch.tensor([X[0]])
x_1 = torch.tensor([X[1]])

out = model(inputs = x, conditional_tokens = {"conditional_tokens_1": x_1})
out.shape
```



# Overall Structure For Cell Representation

The primary object that will prepare the cell_representations is `Heimdall.cell_representations.Cell_Representation`. A minimal example is provided in both `main.py` and `demo.ipynb` that showcases how to use hydra and omegaConf and how to prepare `f_g` and `f_c` accordingly.

## Quickstart For Cell Representation

```
fg = identity_fg
fc = geneformer_fc

CR = Cell_Representation(config) ## takes in the whole config from hydra
CR.preprocess_anndata() ## standard sc preprocessing can be done here
CR.preprocess_f_g(identity_fg) ## takes in the identity f_g specified above
CR.preprocess_f_c(geneformer_fc) ## takes in the geneformer f_c specified above
CR.prepare_labels() ## prepares the labels

## we can take this out here now and pass this into a PyTorch dataloader and separately create the model
X = CR.cell_representation
y = CR.labels
```



# Gene Representation: $f_g$

This is quite fluid, but in general, the `f_g` function prepared takes in an `anndata.var` dataframe, and reprocesses the gene names into a dictionary that maps gene names to either (1) integers, (2) lists or (3) numpy arrays. 

Here is an example that does it only for integers:

```
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
```

This is activated by passing it through the function:

```
CR.preprocess_f_g(identity_fg) ## takes in the identity f_g specified above
```


# CellxGene Representation: $f_c$

This is a lot more open ended, in general, the `fc` will always take in the `fg` (gene name to integer/vector dictionary) and 
the anndata object. 

The `fc` will return a list, where each entry is a single cell example. In this example below `dataset[0]` is an `n_gene x 1` 
vector denoting one cell as each entry is a gene id. 


```
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
```


This is activated by passing it through the function:

```
CR.preprocess_f_c(geneformer_fc) ## takes in the geneformer f_c specified above
```
