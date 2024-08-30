[![Lint](https://github.com/gkrieg/Heimdall/actions/workflows/lint.yml/badge.svg)](https://github.com/gkrieg/Heimdall/actions/workflows/lint.yml)

# Heimdall

# Installation

```bash
git clone

cd Heimdall

conda create --name heimdall python=3.10
conda activate heimdall
pip install -r requirements.txt
pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Quickstart

`train.py` provides a clear overview of the inputs needed, how to prepare the data, model, optimizer, and run the trainer.

```
python train.py +experiments=classification_experiment_dev
```

Make sure to edit the global file `config/global_vars.yaml` based on your set up.

# Sweeps

`scripts/create_sweep.py`  has the arguments `--experiment-name` (the hydra experiment file name),  `--project-name` (W&B project name), `--fg` and `--fc` which are the names of the hydra configs. It is a short script that will load in `sweeps/base.yaml` and updates it appropriately, and creates a sweep argument and returns it. This can work in tandem with `deploy_sweep.sh` to submit multiple sweeps on SLURM systems.

```
python scripts/create_sweep.py --experiment-name pancreas --project-name Pancreas-Celltype-Classification
```

# Heimdall Trainer Documentation (outdated)

The `Heimdall_Trainer` object now will automatically consider and process the training with Huggingface Accelerate

```python
#####
# Initialize the Trainer
#####
trainer = Heimdall_Trainer(config=config, model=model, optimizer=optimizer,
            dataloader_train = dataloader_train,
            dataloader_val = dataloader_val,
            dataloader_test = dataloader_test,
            run_wandb = True)


### Training
trainer.train()
```

# Model Documentation

\[Deprecated\] `TransformerConfig` is deprecated and has been integrated into the hydra config.

The `Heimdall_Transformer` object is a default transformer that is flexible for `learned` embeddings and `predefined` embeddings, as well as conditional tokens that can be `learned` or `predefined` as well. Here is an example:

```python
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

```python
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

```python
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

```python
CR.preprocess_f_g(identity_fg) ## takes in the identity f_g specified above
```

# CellxGene Representation: $f_c$

This is a lot more open ended, in general, the `fc` will always take in the `fg` (gene name to integer/vector dictionary) and
the anndata object.

The `fc` will return a list, where each entry is a single cell example. In this example below `dataset[0]` is an `n_gene x 1`
vector denoting one cell as each entry is a gene id.

```python
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

```python
CR.preprocess_f_c(geneformer_fc) ## takes in the geneformer f_c specified above
```

# Dev Notes

## Dev installation

```bash
pip install -r requirements.txt
```

Once the `pre-commit` command line tool is installed, every time you commit
some changes, it will perform several code-style checks and automatically
apply some fixes for you (if there is any issue). When auto-fixes
are applied, you need to recommit those changes. Note that this process can
take more than one round.

After you are done committing changes and are ready to push the commits to the
remote branch, run `nox` to perform a final quality check. Note that `nox` is
linting only and does not fix the issues for you. You need to address
the issues manually based on the instructions provided.

```bash
nox
```

Commit changes once the quality checks are passed, which triggers the pre-commit
hook to run some final formatting checks and catch any remaining issues. Note
that the pre-commit hooks could automatically apply fixes to the current
commit. In that case, you can first review the changes and accept them if they
are appropriate, or make alternative changes to suppress the error. Afterwards,
recommit the changes.

## Cheatsheet

```bash
# Run cell type classification dev experiment with wandb disabled
WANDB_MODE=disabled python train.py +experiments=classification_experiment_dev

# Run cell type classification dev experiment with wandb offline mode
WANDB_MODE=offline python train.py +experiments=classification_experiment_dev

# Run cell cell interaction dev experiment with wandb disabled
WANDB_MODE=disabled python train.py +experiments=cell_cell_interaction_dev

# Run cell cell interaction dev experiment with wandb disabled and overwrite epochs
WANDB_MODE=disabled python train.py +experiments=cell_cell_interaction_dev tasks.args.epochs=2

# Run cell cell interaction dev experiment with user profile (dev has wandb disabled by default)
python train.py +experiments=cell_cell_interaction_dev user=lane-remy-dev
```

## Local tests

We use [pytest](https://docs.pytest.org/en/stable/getting-started.html) to write local tests.
New test suites can be added under `tests/test_{suite_name}.py`.

Run a particular test suite with:

```bash
python -m pytest tests/test_{suite_name}.py
```
