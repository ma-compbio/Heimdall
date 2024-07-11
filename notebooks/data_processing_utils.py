from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mygene

MG = mygene.MyGeneInfo()


@dataclass
class GeneMappingOutput:
    """Gene mapping results data structure.

    Attributes:
        mapping_full: Dictionary mapping from query gene id to target ids as
            a list. If the mapping is not available, then this gene id would
            not appear in this mapping. Thus, this dictionary mmight contain
            less elements than the total number of the queried genes.
        mapping_combined: Dictionary mapping from query gene id to combined
            target ids (concatenated by "|", e.g., ["id1", "id2"] would be
            converted to "id1|id2"). If the gene id conversion is not
            applicable, then we will map it to "N/A". Thus, this dictionary
            contains the same number of element as the total number of the
            queried genes.
        mapping_reduced: Similar to mapping_combined, but only use the first
            mapped ids (sorted alphabetically) when multiple ids are available.
            Furthermore, use the query gene id as the target gene id if the
            mapping is unavailable. This dictionary contains the same number of
            elements as the total number of the queried genes.

    """
    mapping_full: Dict[str, List[str]]
    mapping_combined: Dict[str, str]
    mapping_reduced: Dict[str, str]


def symbol_to_ensembl(
    genes: List[str],
    species: str = "human",
    extra_query_kwargs: Optional[Dict[str, Any]] = None,
) -> GeneMappingOutput:
    # Query from MyGene.Info server
    print(f"Querying {len(genes):,} genes")
    query_results = MG.querymany(
        genes,
        species=species,
        scopes="symbol",
        fields="ensembl.gene",
        **(extra_query_kwargs or {}),
    )

    # Unpack query results
    symbol_to_ensembl = {}
    for res in query_results:
        symbol = res["query"]
        if (ensembl := res.get("ensembl")) is None:
            continue

        if isinstance(ensembl, dict):
            new_ensembl_genes = [ensembl["gene"]]
        elif isinstance(ensembl, list):
            new_ensembl_genes = [i["gene"] for i in ensembl]
        else:
            raise ValueError(f"Unknown ensembl query result type {type(ensembl)}: {ensembl!r}")

        symbol_to_ensembl[symbol] = symbol_to_ensembl.get(symbol, []) + new_ensembl_genes

    # Consolidate
    symbol_to_ensembl_combined = {}
    symbol_to_ensembl_reduced = {}
    for symbol in genes:
        if (ensembl := symbol_to_ensembl.get(symbol)) is None:
            symbol_to_ensembl_combined[symbol] = "N/A"
            symbol_to_ensembl_reduced[symbol] = symbol
        else:
            ensembl = sorted(set(ensembl))
            symbol_to_ensembl[symbol] = ensembl
            symbol_to_ensembl_reduced[symbol] = ensembl[0]
            symbol_to_ensembl_combined[symbol] = "|".join(ensembl)

    print(
        f"Successfully mapped {len(symbol_to_ensembl):,} out of "
        f"{len(genes):,} genes ({len(symbol_to_ensembl) / len(genes):.1%})",
    )

    return GeneMappingOutput(symbol_to_ensembl, symbol_to_ensembl_combined, symbol_to_ensembl_reduced)
