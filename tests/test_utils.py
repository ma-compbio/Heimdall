from pathlib import Path

import awkward as ak
import numpy as np

from Heimdall.utils import searchsorted2d, symbol_to_ensembl, symbol_to_ensembl_from_ensembl


def test_searchsorted2d():
    num_features = 4
    values = np.arange(5 * num_features).reshape((5, num_features))

    quantiles = np.linspace(0, 1, num_features)
    bin_edges = ak.Array(
        np.quantile(values, quantiles, axis=1).T,
    )
    result = searchsorted2d(bin_edges, values)
    expected = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )

    assert np.allclose(expected, result)


def test_awkward_searchsorted2d():
    values = ak.Array(
        [
            [0, 1, 2],
            [3, 4, 5, 6],
            [7, 8],
            [9],
        ],
    )

    bin_edges = ak.Array([np.quantile(row, np.linspace(0, 1, len(row))) for row in values])
    result = searchsorted2d(bin_edges, values)
    expected = ak.Array(
        [
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1],
            [0],
        ],
    )

    for expected_row, result_row in zip(expected, result):
        assert np.allclose(expected_row, result_row)


def test_symbol_to_ensembl(request):
    species = "human"
    gene_table_symbol_to_ensembl = {
        "PPIEL": ["ENSG00000243970", "ENSG00000291129"],
        "PRDM16": ["ENSG00000142611"],
        "PEX10": ["ENSG00000157911"],
        "RNA5-8SN5": ["ENSG00000274917"],
        "DOESNOTEXIST": None,
    }
    genes = list(gene_table_symbol_to_ensembl)
    cache_data_dir = Path(request.config.cache.makedir("data"))

    def check_result(res):
        print(res.mapping_full)
        assert res.genes == genes
        for g in genes:
            if gene_table_symbol_to_ensembl[g] is None:
                assert g not in res.mapping_full
                assert res.mapping_combined[g] == "N/A"
                assert res.mapping_reduced[g] == g
            else:
                assert res.mapping_full[g] == gene_table_symbol_to_ensembl[g]
                assert res.mapping_combined[g] == "|".join(gene_table_symbol_to_ensembl[g])
                assert res.mapping_reduced[g] == gene_table_symbol_to_ensembl[g][0]
        print("Success!\n")

    check_result(symbol_to_ensembl(genes, species=species))
    check_result(symbol_to_ensembl_from_ensembl(data_dir=cache_data_dir, genes=genes, species=species))
