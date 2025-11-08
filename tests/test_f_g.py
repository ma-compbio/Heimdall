from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from omegaconf import OmegaConf
from pytest import fixture

from Heimdall.fg import CSVFg, IdentityFg, TorchTensorFg


def test_identity_fg(mock_dataset, identity_fg):
    gene_names = mock_dataset.adata.var_names

    identity_fg.preprocess_embeddings()

    embedding_indices = identity_fg[gene_names]
    assert np.allclose(embedding_indices, np.arange(len(gene_names)))
    assert identity_fg.pad_value == 4
    assert identity_fg.mask_value == 5


def test_torch_tensor_fg(mock_dataset):
    config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "constructor": "from_pretrained",
                "args": {
                    "embeddings": "gene_embeddings",
                },
            },
            "vocab_size": 6,
            "d_embedding": 128,
            "embedding_filepath": Path(
                "/work/magroup/shared/Heimdall/data/pretrained_embeddings/ESM2/protein_map_human_ensembl.pt",
            ),
        },
    )
    if not config.embedding_filepath.is_file():
        pytest.skip(f"Skipping due to missing file {config.embedding_filepath}")

    gene_names = mock_dataset.adata.var_names
    valid_gene_mask = [gene_name != "fake_gene" for gene_name in gene_names]

    expected_valid_values = [0.04376760497689247, 0.1535314917564392, 0.11875522881746292]

    esm2_fg = TorchTensorFg(mock_dataset, **config)
    esm2_fg.preprocess_embeddings()

    try:
        embedding_indices = esm2_fg[gene_names]
    except KeyError:
        pass

    embedding_indices = esm2_fg[gene_names[valid_gene_mask]]
    embeddings = esm2_fg.gene_embeddings[embedding_indices]
    assert np.allclose(embeddings[:, 0], expected_valid_values)

    assert esm2_fg.pad_value == 3
    assert esm2_fg.mask_value == 4


def test_csv_fg(mock_dataset):
    config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "constructor": "from_pretrained",
                "args": {
                    "embeddings": "gene_embeddings",
                },
            },
            "vocab_size": 6,
            "d_embedding": 128,
            "embedding_filepath": Path(
                "/work/magroup/shared/Heimdall/data/pretrained_embeddings/gene2vec/gene2vec_genes.txt",
            ),
        },
    )
    if not config.embedding_filepath.is_file():
        pytest.skip(f"Skipping due to missing file {config.embedding_filepath}")

    gene_names = mock_dataset.adata.var_names
    valid_gene_mask = [gene_name != "fake_gene" for gene_name in gene_names]
    expected_valid_values = [0.09901640564203262, -0.02311580441892147, 0.33965930342674255]

    gene2vec_fg = CSVFg(mock_dataset, **config)
    gene2vec_fg.preprocess_embeddings()

    try:
        embedding_indices = gene2vec_fg[gene_names]
    except KeyError:
        pass

    embedding_indices = gene2vec_fg[gene_names[valid_gene_mask]]
    embeddings = gene2vec_fg.gene_embeddings[embedding_indices]
    assert np.allclose(embeddings[:, 0], expected_valid_values)
    assert gene2vec_fg.pad_value == 3
    assert gene2vec_fg.mask_value == 4
