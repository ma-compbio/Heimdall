import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mygene
import pandas as pd
import requests
from tqdm.auto import tqdm

MG = mygene.MyGeneInfo()

ENSEMBL_URL_MAP = {
    "human": "https://ftp.ensembl.org/pub/release-{}/gtf/homo_sapiens/Homo_sapiens.GRCh38.{}.gtf.gz",
    "mouse": "https://ftp.ensembl.org/pub/release-{}/gtf/mus_musculus/Mus_musculus.GRCm39.{}.gtf.gz",
}


@dataclass
class GeneMappingOutput:
    """Gene mapping results data structure.

    Attributes:
        genes: List of original gene ids.
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
    genes: List[str]
    mapping_full: Dict[str, List[str]]
    mapping_combined: Dict[str, str] = field(init=False)
    mapping_reduced: Dict[str, str] = field(init=False)

    def __post_init__(self):
        self.mapping_combined = {}
        self.mapping_reduced = {}
        for g in self.genes:
            if (ensembl := self.mapping_full.get(g)) is None:
                self.mapping_combined[g] = "N/A"
                self.mapping_reduced[g] = g
            else:
                ensembl = sorted(set(ensembl))
                self.mapping_full[g] = ensembl
                self.mapping_reduced[g] = ensembl[0]
                self.mapping_combined[g] = "|".join(ensembl)

        print(
            f"Successfully mapped {len(self.mapping_full):,} out of {len(self.genes):,} "
            f"genes ({len(self.mapping_full) / len(self.genes):.1%})",
        )


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
    symbol_to_ensembl_dict = {}
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

        symbol_to_ensembl_dict[symbol] = symbol_to_ensembl_dict.get(symbol, []) + new_ensembl_genes

    return GeneMappingOutput(genes, symbol_to_ensembl_dict)


def symbol_to_ensembl_from_ensembl(
    data_dir: Union[str, Path],
    genes: List[str],
    species: str = "human",
    release: int = 112,
) -> GeneMappingOutput:
    mapping_dict = _load_ensembl_table(data_dir, species, release)
    symbol_to_ensembl_dict = {i: mapping_dict[i] for i in genes if i in mapping_dict}
    return GeneMappingOutput(genes, symbol_to_ensembl_dict)


def _load_ensembl_table(
    data_dir: Union[str, Path],
    species: str,
    release: int,
) -> Dict[str, List[str]]:
    try:
        url = ENSEMBL_URL_MAP[species].format(release, release)
        fname = url.split("/")[-1]
    except KeyError as e:
        raise KeyError(
            f"Unknown species {species!r}, available options are {sorted(ENSEMBL_URL_MAP)}",
        ) from e

    data_dir = Path(data_dir).resolve() / "gene_mapping" / "ensembl" / species
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mapping data directory: {data_dir}")

    raw_path = data_dir / fname
    attr_path = data_dir / "gene_attr_table.tsv.gz"
    mapping_symbol_to_ensembl_path = data_dir / "symbol_to_ensembl.json"
    mapping_ensembl_to_symbol_path = data_dir / "ensembl_to_symbol.json"

    # Download GTF from Ensembl
    if not raw_path.is_file():
        print(f"Downloading gene annotation from {url}")
        with requests.get(url) as r:
            if not r.ok:
                raise requests.RequestException(f"Fail to download {url} ({r})")

            with open(raw_path, "wb") as f:
                f.write(r.content)

    # Prepare gene attribute table from GTF
    if not attr_path.is_file():
        print("Extracting gene attributes")
        full_df = pd.read_csv(raw_path, sep="\t", compression="gzip", comment="#", header=None, low_memory=False)

        def flat_to_dict(x: str) -> Tuple[str, str]:
            """Convert a flat string to a dictionary.

            Example:
                'gene_id "xxx"; gene_name "ABC";' -> {"gene_id": "xxx", "gene_name": "ABC"}

            """
            data = []
            for i in x.split("; "):
                key, val = i.split(" ", 1)
                data.append((key.strip('"'), val.rstrip(';').strip('"')))
            return dict(data)

        attr_df = pd.DataFrame(list(map(flat_to_dict, tqdm(full_df[8]))))
        attr_df.to_csv(attr_path, sep="\t", index=False, compression="gzip")

    # Prepare symbol-ensembl mappings
    if not mapping_symbol_to_ensembl_path.is_file():
        print("Preparing gene ID mapping")
        attr_df = pd.read_csv(attr_path, sep="\t", compression="gzip")
        mapping_df = attr_df[["gene_id", "gene_name"]].drop_duplicates()

        symbol_to_ensembl, ensembl_to_symbol = defaultdict(list), defaultdict(list)
        for _, (ensembl, symbol) in mapping_df.iterrows():
            symbol_to_ensembl[symbol].append(ensembl)
            ensembl_to_symbol[ensembl].append(symbol)

        symbol_to_ensembl, ensembl_to_symbol = map(
            lambda x: {i: sorted(j) for i, j in x.items()},
            (symbol_to_ensembl, ensembl_to_symbol),
        )

        with (
            open(mapping_symbol_to_ensembl_path, "w") as f1,
            open(mapping_ensembl_to_symbol_path, "w") as f2,
        ):
            json.dump(symbol_to_ensembl, f1, indent=4)
            json.dump(ensembl_to_symbol, f2, indent=4)
        print(f"Mapping saved to cache: {mapping_symbol_to_ensembl_path}")

    else:
        print(f"Loading mapping from cache: {mapping_symbol_to_ensembl_path}")
        with open(mapping_symbol_to_ensembl_path) as f:
            symbol_to_ensembl = json.load(f)

    return symbol_to_ensembl


if __name__ == "__main__":
    species = "human"
    gene_table_symbol_to_ensembl = {
        "PPIEL": ["ENSG00000243970", "ENSG00000291129"],
        "PRDM16": ["ENSG00000142611"],
        "PEX10": ["ENSG00000157911"],
        "RNA5-8SN5": ["ENSG00000274917"],
        "DOESNOTEXIST": None,
    }
    genes = list(gene_table_symbol_to_ensembl)

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
    check_result(symbol_to_ensembl_from_ensembl(data_dir="dev_test", genes=genes, species=species))
