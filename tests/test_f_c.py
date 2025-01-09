from pathlib import Path

import anndata as ad
import numpy as np
from omegaconf import OmegaConf
from pytest import fixture
from scipy.sparse import csr_array

from Heimdall.fc import GeneformerFc, ScGPTFc


@fixture
def geneformer_fc(zero_expression_mock_dataset, zero_expression_identity_fg, zero_expression_sorting_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 4,
        },
    )
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_sorting_fe.preprocess_embeddings()

    geneformer_fc = GeneformerFc(
        zero_expression_identity_fg,
        zero_expression_sorting_fe,
        zero_expression_mock_dataset,
        **fc_config,
    )

    return geneformer_fc


@fixture
def scgpt_fc(zero_expression_mock_dataset, zero_expression_identity_fg, zero_expression_binning_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 2,
        },
    )
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_binning_fe.preprocess_embeddings()

    scgpt_fc = ScGPTFc(
        zero_expression_identity_fg,
        zero_expression_binning_fe,
        zero_expression_mock_dataset,
        **fc_config,
    )

    return scgpt_fc


def test_geneformer_fc_preprocess_cells_and_getitem(zero_expression_mock_dataset, geneformer_fc):
    identity_expected = csr_array(
        np.array(
            [
                [1, 2, 3],
                [2, 3, 0],
                [3, 0, 1],
                [0, 1, 2],
            ],
        ),
    )

    _, raw_seq_length = identity_expected.shape

    for cell_index in range(len(zero_expression_mock_dataset)):
        identity_inputs, expression_inputs, padding_mask = geneformer_fc[cell_index]
        assert np.allclose(identity_expected[[cell_index], :].toarray(), identity_inputs[:raw_seq_length])
        assert len(identity_inputs) == geneformer_fc.max_input_length

        assert not np.any(padding_mask[:raw_seq_length])
        assert np.all(padding_mask[raw_seq_length:])


def test_scgpt_fc_preprocess_cells_and_getitem(zero_expression_mock_dataset, scgpt_fc):
    identity_expected = csr_array(
        np.array(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
        ),
    )

    expression_expected = csr_array(
        np.array(
            [
                [3, 2, 1],
                [1, 3, 2],
                [2, 1, 3],
                [3, 2, 1],
            ],
        ),
    )

    _, raw_seq_length = identity_expected.shape

    seed = 0
    rng = np.random.default_rng(seed)
    for cell_index in range(len(zero_expression_mock_dataset)):
        identity_inputs, expression_inputs, padding_mask = scgpt_fc[cell_index]

        sample_indices = rng.choice(raw_seq_length, scgpt_fc.max_input_length, replace=False)
        assert np.allclose(identity_expected[[cell_index], sample_indices], identity_inputs)
        assert np.allclose(expression_expected[[cell_index], sample_indices], expression_inputs)
        assert len(identity_inputs) == scgpt_fc.max_input_length

        assert not np.any(padding_mask[: scgpt_fc.max_input_length])


def test_geneformer_fc_embed_cells(geneformer_fc):
    ...
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
