from pathlib import Path

from Heimdall.utils import symbol_to_ensembl, symbol_to_ensembl_from_ensembl


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
