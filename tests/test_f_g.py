from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from omegaconf import OmegaConf
from pytest import fixture

from Heimdall.fg import ESM2Fg, Gene2VecFg, IdentityFg


@fixture
def mock_dataset():
    gene_names = ["ENSG00000121410", "ENSG00000148584", "fake_gene", "ENSG00000175899"]

    mock_expression = np.array(
        [
            [1, 4, 3, 2],
            [2, 1, 4, 3],
            [3, 2, 1, 4],
            [4, 3, 2, 1],
        ],
    )

    mock_dataset = ad.AnnData(X=mock_expression)
    mock_dataset.var_names = gene_names

    return mock_dataset


def test_identity_fg(mock_dataset):
    config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "args": {
                    "num_embeddings": "vocab_size",
                    "out_features": "128",
                },
            },
            "vocab_size": 6,
            "d_embedding": 128,
        },
    )

    gene_names = mock_dataset.var_names

    identity_fg = IdentityFg(mock_dataset, **config)
    identity_fg.preprocess_embeddings()

    embedding_indices = identity_fg[gene_names]
    assert np.allclose(embedding_indices, np.arange(len(gene_names)))


def test_esm2_fg(mock_dataset):
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

    gene_names = mock_dataset.var_names
    valid_gene_mask = [gene_name != "fake_gene" for gene_name in gene_names]

    expected_valid_values = [0.04376760497689247, 0.1535314917564392, 0.11875522881746292]

    esm2_fg = ESM2Fg(mock_dataset, **config)
    esm2_fg.preprocess_embeddings()

    try:
        embedding_indices = esm2_fg[gene_names]
    except KeyError:
        pass

    embedding_indices = esm2_fg[gene_names[valid_gene_mask]]
    embeddings = esm2_fg.gene_embeddings[embedding_indices]
    assert np.allclose(embeddings[:, 0], expected_valid_values)


def test_gene2vec_fg(mock_dataset):
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

    gene_names = mock_dataset.var_names
    valid_gene_mask = [gene_name != "fake_gene" for gene_name in gene_names]
    expected_valid_values = [0.09901640564203262, -0.02311580441892147, 0.33965930342674255]

    gene2vec_fg = Gene2VecFg(mock_dataset, **config)
    gene2vec_fg.preprocess_embeddings()

    try:
        embedding_indices = gene2vec_fg[gene_names]
    except KeyError:
        pass

    embedding_indices = gene2vec_fg[gene_names[valid_gene_mask]]
    embeddings = gene2vec_fg.gene_embeddings[embedding_indices]
    assert np.allclose(embeddings[:, 0], expected_valid_values)
