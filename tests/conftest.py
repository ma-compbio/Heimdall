import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pytest import fixture
from scipy.sparse import csr_array

from Heimdall.fe import BinningFe, DummyFe, NonzeroIdentityFe, SortingFe
from Heimdall.fg import IdentityFg


@fixture(scope="module")
def plain_toy_data():
    return ad.AnnData(
        X=csr_array(np.arange(3 * 5).reshape(5, 3)),
        var=pd.DataFrame(index=["ENSG00000142611", "ENSG00000157911", "ENSG00000274917"]),
    )


@fixture(scope="module")
def toy_single_data_path(pytestconfig, plain_toy_data):
    data_path = pytestconfig.cache.mkdir("toy_data")

    adata = plain_toy_data.copy()
    adata.obs["split"] = "train"
    adata.obs["class"] = 0

    path = data_path / "toy_single_adata.h5ad"
    adata.write_h5ad(path)

    return path


@fixture(scope="module")
def toy_paried_data_path(pytestconfig, plain_toy_data):
    data_path = pytestconfig.cache.mkdir("toy_data")

    adata = plain_toy_data.copy()
    zeros = csr_array((adata.shape[0], adata.shape[0]))
    for i, key in enumerate(("train", "val", "test", "task")):
        adata.obsp[key] = zeros.copy()
        if key != "task":
            adata.obsp[key][i, i] = 1

    path = data_path / "toy_single_adata.h5ad"
    adata.write_h5ad(path)

    return path


@fixture
def mock_dataset():
    gene_names = ["ENSG00000121410", "ENSG00000148584", "fake_gene", "ENSG00000175899"]

    mock_expression = csr_array(
        np.array(
            [
                [1, 4, 3, 2],
                [2, 1, 4, 3],
                [3, 2, 1, 4],
                [4, 3, 2, 1],
            ],
        ),
    )

    mock_dataset = ad.AnnData(X=mock_expression)
    mock_dataset.var_names = gene_names

    return mock_dataset


@fixture
def zero_expression_mock_dataset():
    gene_names = ["ENSG00000121410", "ENSG00000148584", "fake_gene", "ENSG00000175899"]

    mock_expression = csr_array(
        np.array(
            [
                [0, 3, 2, 1],
                [1, 0, 3, 2],
                [2, 1, 0, 3],
                [3, 2, 1, 0],
            ],
        ),
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
                "type": "Heimdall.embedding.FlexibleTypeLinear",
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
def dummy_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": 6,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
            "d_embedding": None,
        },
    )
    dummy_fe = DummyFe(mock_dataset, **fe_config)

    return dummy_fe


@fixture
def nonzero_identity_fe(zero_expression_mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": 6,
            "embedding_parameters": {
                "type": "Heimdall.embedding.TwoLayerNN",
                "args": {
                    "in_features": 1,
                    "out_features": 128,
                },
            },
            "d_embedding": 128,
        },
    )
    nonzero_identity_fe = NonzeroIdentityFe(zero_expression_mock_dataset, **fe_config)

    return nonzero_identity_fe


@fixture
def zero_expression_binning_fe(zero_expression_mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": 6,
            "embedding_parameters": {
                "type": "Heimdall.embedding.FlexibleTypeLinear",
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
