from pathlib import Path

import anndata as ad
import awkward as ak
import numpy as np
from omegaconf import OmegaConf
from pytest import fixture

from Heimdall.fe import BinningFe, DummyFe, SortingFe
from Heimdall.fg import IdentityFg


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


@fixture
def zero_expression_mock_dataset():
    gene_names = ["ENSG00000121410", "ENSG00000148584", "fake_gene", "ENSG00000175899"]

    mock_expression = np.array(
        [
            [0, 3, 2, 1],
            [1, 0, 3, 2],
            [2, 1, 0, 3],
            [3, 2, 1, 0],
        ],
    )

    mock_dataset = ad.AnnData(X=mock_expression)
    mock_dataset.var_names = gene_names

    return mock_dataset


@fixture
def identity_fg(mock_dataset):
    fg_config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "args": {
                    "num_embeddings": "vocab_size",
                    "embedding_dim": 128,
                },
            },
            "vocab_size": 6,
            "d_embedding": 128,
        },
    )
    identity_fg = IdentityFg(mock_dataset, **fg_config)

    return identity_fg


@fixture
def zero_expression_identity_fg(zero_expression_mock_dataset):
    fg_config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "args": {
                    "num_embeddings": "vocab_size",
                    "embedding_dim": 128,
                },
            },
            "vocab_size": 6,
            "d_embedding": 128,
        },
    )
    identity_fg = IdentityFg(zero_expression_mock_dataset, **fg_config)

    return identity_fg


@fixture
def sorting_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "args": {
                    "num_embeddings": "vocab_size",
                    "embedding_dim": 128,
                },
            },
            "vocab_size": 6,
            "d_embedding": 128,
        },
    )
    sorting_fe = SortingFe(mock_dataset, **fe_config)

    return sorting_fe


@fixture
def zero_expression_sorting_fe(zero_expression_mock_dataset):
    fe_config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "args": {
                    "num_embeddings": "vocab_size",
                    "embedding_dim": 128,
                },
            },
            "vocab_size": 6,
            "d_embedding": 128,
        },
    )
    sorting_fe = SortingFe(zero_expression_mock_dataset, **fe_config)

    return sorting_fe


@fixture
def binning_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": int(np.max(mock_dataset.X)) + 2,
            "embedding_parameters": {
                "type": "Heimdall.utils.FlexibleTypeLinear",
                "args": {
                    "in_features": 1,  # Replace later
                    "out_features": 128,
                },
            },
            "d_embedding": 128,
            "num_bins": int(np.max(mock_dataset.X)),
        },
    )
    binning_fe = BinningFe(mock_dataset, **fe_config)

    return binning_fe


@fixture
def zero_expression_binning_fe(zero_expression_mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": 6,
            "embedding_parameters": {
                "type": "Heimdall.utils.FlexibleTypeLinear",
                "args": {
                    "in_features": 1,  # Replace later
                    "out_features": 128,
                },
            },
            "d_embedding": 128,
            "num_bins": int(np.max(zero_expression_mock_dataset.X)),
        },
    )
    binning_fe = BinningFe(zero_expression_mock_dataset, **fe_config)

    return binning_fe


def test_sorting_fe(identity_fg, sorting_fe):
    identity_fg.preprocess_embeddings()
    sorting_fe.preprocess_embeddings()

    output = sorting_fe.adata.obsm["processed_expression_values"]

    _, num_genes = sorting_fe.adata.shape

    final_output = np.asarray(ak.fill_none(ak.pad_none(output, num_genes), -1))

    expected = np.array(
        [
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 1, 2, 3],
        ],
    )

    assert np.allclose(expected, output)

    assert sorting_fe.pad_value == 4
    assert sorting_fe.mask_value == 5


def test_zero_expression_sorting_fe(zero_expression_identity_fg, zero_expression_sorting_fe):
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_sorting_fe.preprocess_embeddings()

    output = zero_expression_sorting_fe.adata.obsm["processed_expression_values"]

    num_genes = zero_expression_sorting_fe.num_genes

    padded_output = np.asarray(ak.fill_none(ak.pad_none(output, num_genes), -1))

    expected = np.array(
        [
            [1, 2, 3],
            [2, 3, 0],
            [3, 0, 1],
            [0, 1, 2],
        ],
    )

    padded_expected = np.array(
        [
            [1, 2, 3, -1],
            [2, 3, 0, -1],
            [3, 0, 1, -1],
            [0, 1, 2, -1],
        ],
    )

    assert np.allclose(expected, output)
    assert np.allclose(padded_expected, padded_output)

    assert zero_expression_sorting_fe.pad_value == 4
    assert zero_expression_sorting_fe.mask_value == 5


def test_zero_expression_binning_fe(zero_expression_identity_fg, zero_expression_binning_fe):
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_binning_fe.preprocess_embeddings()

    output = zero_expression_binning_fe.adata.obsm["processed_expression_values"]

    num_genes = zero_expression_binning_fe.num_genes

    padded_output = np.asarray(ak.fill_none(ak.pad_none(output, num_genes), -1))

    expected = np.array(
        [
            [3, 2, 1],
            [1, 3, 2],
            [2, 1, 3],
            [3, 2, 1],
        ],
    )

    padded_expected = np.array(
        [
            [3, 2, 1, -1],
            [1, 3, 2, -1],
            [2, 1, 3, -1],
            [3, 2, 1, -1],
        ],
    )

    assert np.allclose(expected, output)
    assert np.allclose(padded_expected, padded_output)

    assert zero_expression_binning_fe.pad_value == zero_expression_binning_fe.num_bins
    assert zero_expression_binning_fe.mask_value == zero_expression_binning_fe.num_bins + 1


def test_binning_fe(identity_fg, binning_fe):
    identity_fg.preprocess_embeddings()
    binning_fe.preprocess_embeddings()

    output = binning_fe.adata.obsm["processed_expression_values"]

    expected = binning_fe.adata.X

    assert np.allclose(expected, output)

    assert binning_fe.pad_value == binning_fe.num_bins
    assert binning_fe.mask_value == binning_fe.num_bins + 1
