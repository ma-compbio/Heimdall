from pathlib import Path

import anndata as ad
import numpy as np
from omegaconf import OmegaConf
from pytest import fixture

from Heimdall.fc import GeneformerFc, ScGPTFc
from Heimdall.fe import BinningFe, SortingFe
from Heimdall.fg import Gene2VecFg, IdentityFg


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
def identity_fg(mock_dataset):
    fg_config = OmegaConf.create(
        {
            "embedding_filepath": None,
            "d_embedding": 128,
        },
    )
    identity_fg = IdentityFg(mock_dataset, **fg_config)

    return identity_fg


@fixture
def sorting_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "embedding_filepath": None,
            "num_embeddings": None,
            "d_embedding": None,
        },
    )
    sorting_fe = SortingFe(mock_dataset, **fe_config)

    return sorting_fe


@fixture
def binning_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "embedding_filepath": None,
            "num_embeddings": int(np.max(mock_dataset.X)) + 1,
            "d_embedding": 128,
        },
    )
    binning_fe = BinningFe(mock_dataset, **fe_config)

    return binning_fe


@fixture
def geneformer_fc(mock_dataset, identity_fg, sorting_fe):
    fc_config = OmegaConf.create(
        {},
    )
    identity_fg.preprocess_embeddings()
    sorting_fe.preprocess_embeddings()

    geneformer_fc = GeneformerFc(identity_fg, sorting_fe, mock_dataset, **fc_config)

    return geneformer_fc


@fixture
def scgpt_fc(mock_dataset, identity_fg, binning_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 128,
        },
    )
    identity_fg.preprocess_embeddings()
    binning_fe.preprocess_embeddings()

    scgpt_fc = ScGPTFc(identity_fg, binning_fe, mock_dataset, **fc_config)

    return scgpt_fc


def test_geneformer_fc_preprocess_cells(mock_dataset, geneformer_fc):
    geneformer_fc.preprocess_cells()
    geneformer_fc.preprocess_cells()

    output = mock_dataset.obsm["cell_identity_embedding_indices"]

    expected = np.array(
        [
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 1, 2, 3],
        ],
    )

    assert np.allclose(expected, output)


def test_scgpt_fc_preprocess_cells(mock_dataset, scgpt_fc):
    scgpt_fc.preprocess_cells()

    identity_output = mock_dataset.obsm["cell_identity_embedding_indices"]
    identity_expected = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )
    assert np.allclose(identity_expected, identity_output)

    expression_output = mock_dataset.obsm["cell_expression_embedding_indices"]
    expression_expected = mock_dataset.X

    assert np.allclose(expression_expected, expression_output)


def test_geneformer_fc_embed_cells(geneformer_fc):
    geneformer_fc.preprocess_cells()
    # geneformer_fc.embed_cells() # TODO: fill out function call

    # output = mock_dataset.obsm["cell_identity_embedding_indices"]

    # expected = np.array(
    #     [
    #         [1, 2, 3, 0],
    #         [2, 3, 0, 1],
    #         [3, 0, 1, 2],
    #         [0, 1, 2, 3],
    #     ],
    # )

    # assert np.allclose(expected, output)
