from pathlib import Path

import anndata as ad
import awkward as ak
import numpy as np
from omegaconf import OmegaConf
from pytest import fixture
from scipy.sparse import csr_array


def pad(tokens, pad_length, pad_value):
    pad_widths = (0, pad_length - len(tokens))
    padded = np.pad(
        tokens,
        pad_widths,
        "constant",
        constant_values=(pad_value),
    )

    return padded


def test_zero_expression_binning_fe(zero_expression_identity_fg, zero_expression_binning_fe):
    zero_expression_identity_fg.data.set_representation_functions(
        fg=zero_expression_identity_fg,
        fe=zero_expression_binning_fe,
    )

    zero_expression_binning_fe.data.set_representation_functions(
        fg=zero_expression_identity_fg,
        fe=zero_expression_binning_fe,
    )

    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_binning_fe.preprocess_embeddings()

    num_genes = zero_expression_binning_fe.adata.n_vars

    expected = np.array(
        [
            [3, 2, 1],
            [1, 3, 2],
            [2, 1, 3],
            [3, 2, 1],
        ],
    )

    for cell_index in range(len(zero_expression_identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = zero_expression_binning_fe[cell_index]
        assert np.allclose(expected[cell_index], cell_expression_inputs)

    padded_expected = np.array(
        [
            [3, 2, 1, zero_expression_binning_fe.num_bins + 1],
            [1, 3, 2, zero_expression_binning_fe.num_bins + 1],
            [2, 1, 3, zero_expression_binning_fe.num_bins + 1],
            [3, 2, 1, zero_expression_binning_fe.num_bins + 1],
        ],
    )
    for cell_index in range(len(zero_expression_identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = zero_expression_binning_fe[cell_index]
        padded_input = pad(cell_expression_inputs, num_genes, zero_expression_binning_fe.pad_value)
        assert np.allclose(padded_expected[cell_index], padded_input)

    assert zero_expression_binning_fe.pad_value == zero_expression_binning_fe.num_bins + 1
    assert zero_expression_binning_fe.mask_value == zero_expression_binning_fe.num_bins + 2


def test_binning_fe(identity_fg, binning_fe):
    identity_fg.data.set_representation_functions(
        fg=identity_fg,
        fe=binning_fe,
    )

    binning_fe.data.set_representation_functions(
        fg=identity_fg,
        fe=binning_fe,
    )

    identity_fg.preprocess_embeddings()
    binning_fe.preprocess_embeddings()

    expected = binning_fe.adata.X

    for cell_index in range(len(identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = binning_fe[cell_index]
        assert np.allclose(expected[[cell_index], :].toarray(), cell_expression_inputs)

    assert binning_fe.pad_value == binning_fe.num_bins + 1
    assert binning_fe.mask_value == binning_fe.num_bins + 2


def test_scbert_binning_fe(identity_fg, zero_expression_scbert_binning_fe):
    identity_fg.data.set_representation_functions(
        fg=identity_fg,
        fe=zero_expression_scbert_binning_fe,
    )

    zero_expression_scbert_binning_fe.data.set_representation_functions(
        fg=identity_fg,
        fe=zero_expression_scbert_binning_fe,
    )

    identity_fg.preprocess_embeddings()
    zero_expression_scbert_binning_fe.preprocess_embeddings()

    expected = csr_array(
        np.array(
            [
                [2, 2, 1],
                [1, 2, 2],
                [2, 1, 2],
                [2, 2, 1],
            ],
        ),
    )

    for cell_index in range(len(identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = zero_expression_scbert_binning_fe[cell_index]
        assert np.allclose(expected[[cell_index], :].toarray(), cell_expression_inputs)

    assert zero_expression_scbert_binning_fe.pad_value == zero_expression_scbert_binning_fe.num_bins + 1
    assert zero_expression_scbert_binning_fe.mask_value == zero_expression_scbert_binning_fe.num_bins + 2


def test_identity_fe(zero_expression_identity_fg, zero_expression_identity_fe):
    zero_expression_identity_fg.data.set_representation_functions(
        fg=zero_expression_identity_fg,
        fe=zero_expression_identity_fe,
    )

    zero_expression_identity_fe.data.set_representation_functions(
        fg=zero_expression_identity_fg,
        fe=zero_expression_identity_fe,
    )

    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_identity_fe.preprocess_embeddings()

    num_genes = zero_expression_identity_fe.adata.n_vars

    expression_expected = np.array(
        [
            [3, 2, 1],
            [1, 3, 2],
            [2, 1, 3],
            [3, 2, 1],
        ],
    )

    identity_expected = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ],
    )

    for cell_index in range(len(zero_expression_identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = zero_expression_identity_fe[cell_index]
        assert np.allclose(identity_expected[cell_index], cell_identity_inputs)
        assert np.allclose(expression_expected[cell_index], cell_expression_inputs)

    padded_expected = np.array(
        [
            [3, 2, 1, 4],
            [1, 3, 2, 4],
            [2, 1, 3, 4],
            [3, 2, 1, 4],
        ],
    )
    for cell_index in range(len(zero_expression_identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = zero_expression_identity_fe[cell_index]
        padded_input = pad(cell_expression_inputs, num_genes, zero_expression_identity_fe.pad_value)
        assert np.allclose(padded_expected[cell_index], padded_input)

    assert zero_expression_identity_fe.pad_value == 4
    assert zero_expression_identity_fe.mask_value == 5
