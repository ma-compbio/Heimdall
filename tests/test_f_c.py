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
                    "out_features": "128",
                },
            },
            "vocab_size": 6,
            "d_embedding": 128,
        },
    )
    identity_fg = IdentityFg(mock_dataset, **fg_config)

    return identity_fg


@fixture
def sorting_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "args": {
                    "num_embeddings": "vocab_size",
                    "out_features": "128",
                },
            },
            "vocab_size": 6,
            "d_embedding": None,
        },
    )
    sorting_fe = SortingFe(mock_dataset, **fe_config)

    return sorting_fe


@fixture
def binning_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "embedding_parameters": {
                "type": "Heimdall.utils.FlexibleTypeLinear",
                "args": {
                    "in_features": "max_seq_length",
                    "out_features": 128,
                },
            },
            "vocab_size": 6,
            "num_bins": int(np.max(mock_dataset.X)),
            "d_embedding": 128,
            "pad_value": 0,
        },
    )
    binning_fe = BinningFe(mock_dataset, **fe_config)

    return binning_fe


@fixture
def geneformer_fc(mock_dataset, identity_fg, sorting_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 4,
        },
    )
    identity_fg.preprocess_embeddings()
    sorting_fe.preprocess_embeddings()

    geneformer_fc = GeneformerFc(identity_fg, sorting_fe, mock_dataset, **fc_config)

    return geneformer_fc


@fixture
def scgpt_fc(mock_dataset, identity_fg, binning_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 2,
        },
    )
    identity_fg.preprocess_embeddings()
    binning_fe.preprocess_embeddings()

    scgpt_fc = ScGPTFc(identity_fg, binning_fe, mock_dataset, **fc_config)

    return scgpt_fc


def test_geneformer_fc_preprocess_cells_and_getitem(mock_dataset, geneformer_fc):
    geneformer_fc.preprocess_cells()
    geneformer_fc.preprocess_cells()

    output = mock_dataset.obsm["cell_identity_inputs"]

    expected = np.array(
        [
            [1, 2, 3],
            [2, 3, 0],
            [3, 0, 1],
            [0, 1, 2],
        ],
    )

    _, raw_seq_length = expected.shape

    assert np.allclose(expected, output)

    # Get first cell, using padding and truncation
    first_identity_indices, first_expression_indices, first_mask = geneformer_fc[0]

    assert np.allclose(first_identity_indices[:raw_seq_length], first_expression_indices[:raw_seq_length])
    assert np.allclose(first_identity_indices[:raw_seq_length], expected[0])
    assert len(first_identity_indices) == geneformer_fc.max_input_length
    assert not np.any(first_mask[:raw_seq_length])
    assert np.all(first_mask[raw_seq_length:])


def test_scgpt_fc_preprocess_cells_and_getitem(mock_dataset, scgpt_fc):
    scgpt_fc.preprocess_cells()

    identity_output = mock_dataset.obsm["cell_identity_inputs"]
    identity_expected = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ],
    )

    assert np.allclose(identity_expected, identity_output)

    expression_output = mock_dataset.obsm["cell_expression_inputs"]
    expression_expected = np.array(
        [
            [3, 2, 1],
            [1, 3, 2],
            [2, 1, 3],
            [3, 2, 1],
        ],
    )

    assert np.allclose(expression_expected, expression_output)

    # Get first cell, using padding and sampling
    first_identity_indices, first_expression_indices, first_mask = scgpt_fc[0]

    cell_expected = np.stack([identity_expected[0], expression_expected[0]])
    _, input_length = cell_expected.shape

    seed = 0
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(input_length, scgpt_fc.max_input_length, replace=False)

    sampled_first_identity_expected, sampled_first_expression_expected = cell_expected[:, sample_indices]

    assert len(first_identity_indices) == scgpt_fc.max_input_length
    assert np.allclose(first_identity_indices, sampled_first_identity_expected)
    assert np.allclose(first_expression_indices, sampled_first_expression_expected)
    assert not np.any(first_mask)


def test_geneformer_fc_embed_cells(geneformer_fc):
    geneformer_fc.preprocess_cells()
    # geneformer_fc.embed_cells() # TODO: fill out function call

    # output = mock_dataset.obsm["cell_identity_inputs"]

    # expected = np.array(
    #     [
    #         [1, 2, 3, 0],
    #         [2, 3, 0, 1],
    #         [3, 0, 1, 2],
    #         [0, 1, 2, 3],
    #     ],
    # )

    # assert np.allclose(expected, output)
