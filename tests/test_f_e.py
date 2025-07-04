from pathlib import Path

import anndata as ad
import awkward as ak
import numpy as np
from omegaconf import OmegaConf
from pytest import fixture


def pad(tokens, pad_length, pad_value):
    pad_widths = (0, pad_length - len(tokens))
    padded = np.pad(
        tokens,
        pad_widths,
        "constant",
        constant_values=(pad_value),
    )

    return padded


def test_sorting_fe(identity_fg, sorting_fe):
    identity_fg.preprocess_embeddings()
    sorting_fe.preprocess_embeddings()

    # output = sorting_fe.adata.obsm["processed_expression_values"]

    _, num_genes = sorting_fe.adata.shape

    expected = np.array(
        [
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 1, 2, 3],
        ],
    )

    for cell_index in range(len(identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = sorting_fe[cell_index]
        assert np.allclose(expected[cell_index], cell_identity_inputs)

    assert sorting_fe.pad_value == 4
    assert sorting_fe.mask_value == 5


def test_zero_expression_sorting_fe(zero_expression_identity_fg, zero_expression_sorting_fe):
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_sorting_fe.preprocess_embeddings()

    num_genes = zero_expression_sorting_fe.num_genes

    expected = np.array(
        [
            [1, 2, 3],
            [2, 3, 0],
            [3, 0, 1],
            [0, 1, 2],
        ],
    )

    for cell_index in range(len(zero_expression_identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = zero_expression_sorting_fe[cell_index]
        assert np.allclose(expected[cell_index], cell_identity_inputs)

    padded_expected = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 0, 4],
            [3, 0, 1, 4],
            [0, 1, 2, 4],
        ],
    )

    for cell_index in range(len(zero_expression_identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = zero_expression_sorting_fe[cell_index]
        padded_input = pad(cell_identity_inputs, num_genes, zero_expression_sorting_fe.pad_value)
        assert np.allclose(padded_expected[cell_index], padded_input)

    assert zero_expression_sorting_fe.pad_value == 4
    assert zero_expression_sorting_fe.mask_value == 5


def test_zero_expression_binning_fe(zero_expression_identity_fg, zero_expression_binning_fe):
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_binning_fe.preprocess_embeddings()

    num_genes = zero_expression_binning_fe.num_genes

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
    identity_fg.preprocess_embeddings()
    binning_fe.preprocess_embeddings()

    expected = binning_fe.adata.X

    for cell_index in range(len(identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = binning_fe[cell_index]
        assert np.allclose(expected[[cell_index], :].toarray(), cell_expression_inputs)

    assert binning_fe.pad_value == binning_fe.num_bins + 1
    assert binning_fe.mask_value == binning_fe.num_bins + 2


def test_dummy_fe(identity_fg, dummy_fe):
    identity_fg.preprocess_embeddings()
    dummy_fe.preprocess_embeddings()

    expected = dummy_fe.adata.X

    for cell_index in range(len(identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = dummy_fe[cell_index]
        assert np.allclose(expected[[cell_index], :].toarray(), cell_expression_inputs)

    assert dummy_fe.pad_value == 4
    assert dummy_fe.mask_value == 5


def test_nonzero_identity_fe(zero_expression_identity_fg, nonzero_identity_fe):
    zero_expression_identity_fg.preprocess_embeddings()
    nonzero_identity_fe.preprocess_embeddings()

    num_genes = nonzero_identity_fe.num_genes

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
        cell_identity_inputs, cell_expression_inputs = nonzero_identity_fe[cell_index]
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
        cell_identity_inputs, cell_expression_inputs = nonzero_identity_fe[cell_index]
        padded_input = pad(cell_expression_inputs, num_genes, nonzero_identity_fe.pad_value)
        assert np.allclose(padded_expected[cell_index], padded_input)

    assert nonzero_identity_fe.pad_value == 4
    assert nonzero_identity_fe.mask_value == 5


def test_weighted_sampling_fe(identity_fg, weighted_sampling_fe):
    identity_fg.preprocess_embeddings()
    weighted_sampling_fe.preprocess_embeddings()

    # output = weighted_sampling_fe.adata.obsm["processed_expression_values"]

    _, num_genes = weighted_sampling_fe.adata.shape

    expected = np.array(
        [
            [1, 1, 0, 0, 2],
            [2, 2, 2, 1, 2],
            [2, 0, 2, 0, 2],
            [0, 2, 1, 0, 0],
        ],
    )

    for cell_index in range(len(identity_fg.adata)):
        cell_identity_inputs, cell_expression_inputs = weighted_sampling_fe[cell_index]
        assert np.allclose(expected[cell_index], cell_identity_inputs)

    assert weighted_sampling_fe.pad_value == 4
    assert weighted_sampling_fe.mask_value == 5
