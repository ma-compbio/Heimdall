import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytest import fixture
from scipy.sparse import csr_array
from Heimdall.fc import ChromSortRandomSampleFc
from Heimdall.fc import GeneformerFc, ScGPTFc, UCEFc, SortingWeightedResampleFc, SortingTruncateFc, SortingRandomSampleFc
from Heimdall.fe import BinningFe, IdentityFe, SortingFe, WeightedSamplingFe
from Heimdall.fg import IdentityFg
from Heimdall.utils import convert_to_ensembl_ids, instantiate_from_config

load_dotenv()


@fixture(scope="module")
def gene_names():
    return ["A1BG", "A1CF", "fake_gene", "A2M"]


@fixture(scope="module")
def valid_gene_names():
    return ["A1BG", "A1CF", "A2M"]


@fixture(scope="module")
def plain_toy_data(valid_gene_names):
    adata = ad.AnnData(
        X=csr_array(np.arange(3 * 5).reshape(5, 3)),
        var=pd.DataFrame(index=valid_gene_names),
    )

    convert_to_ensembl_ids(adata, data_dir=os.environ["DATA_PATH"])
    return adata


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
def mock_dataset(gene_names):

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
    convert_to_ensembl_ids(mock_dataset, data_dir=os.environ["DATA_PATH"])

    return mock_dataset


@fixture
def mock_dataset_all_valid_genes(valid_gene_names):
    mock_expression = csr_array(
        np.array(
            [
                [1, 4, 2],
                [2, 1, 3],
                [3, 2, 4],
                [4, 3, 1],
            ],
        ),
    )

    mock_dataset = ad.AnnData(X=mock_expression)
    mock_dataset.var_names = valid_gene_names
    convert_to_ensembl_ids(mock_dataset, data_dir=os.environ["DATA_PATH"])

    return mock_dataset


@fixture
def zero_expression_mock_dataset(gene_names):

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
    convert_to_ensembl_ids(mock_dataset, data_dir=os.environ["DATA_PATH"])

    return mock_dataset


@fixture(scope="module")
def chrom_mock_dataset(valid_gene_names, tmp_path_factory):
    """
    4 cells × len(valid_gene_names) genes.
    Here we hard-code chr/start for the three demo genes.
    """
    X = csr_array(
        np.array(
            [
                [1, 4, 2],  # cell 0
                [2, 1, 3],  # cell 1
                [3, 2, 4],  # cell 2
                [4, 3, 1],  # cell 3
            ]
        )
    )

    adata = ad.AnnData(X=X)
    adata.var_names = valid_gene_names
    adata.var["gene_symbol"] = valid_gene_names
    convert_to_ensembl_ids(adata, data_dir=os.environ["DATA_PATH"])

    # Hard-coded chromosome / start info for those genes
    chrom_map  = {"A1BG": "1", "A1CF": "1", "A2M": "2"}
    start_map  = {"A1BG": 100,  "A1CF": 200,  "A2M": 150}

    meta = pd.DataFrame(
        {
            "gene_symbol": valid_gene_names,
            "species": ["human"] * len(valid_gene_names),
            "chromosome": [chrom_map[g] for g in valid_gene_names],
            "start":      [start_map[g] for g in valid_gene_names],
        }
    )
    tmp_dir = tmp_path_factory.mktemp("gene_meta")
    meta_path = tmp_dir / "gene_meta.csv"
    meta.to_csv(meta_path, index=False)

    return adata, meta_path



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
def identity_fg_all_valid_genes(mock_dataset_all_valid_genes):
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
    identity_fg = IdentityFg(mock_dataset_all_valid_genes, **fg_config)

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
def identity_fe(mock_dataset_all_valid_genes):
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
    identity_fe = IdentityFe(mock_dataset_all_valid_genes, **fe_config)

    return identity_fe

@fixture
def zero_expression_identity_fe(zero_expression_mock_dataset):
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
    identity_fe = IdentityFe(zero_expression_mock_dataset, **fe_config)

    return identity_fe



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


@fixture
def weighted_sampling_fe(mock_dataset_all_valid_genes):
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
            "sample_size": 5,
            "d_embedding": 128,
        },
    )
    weighted_sampling_fe = WeightedSamplingFe(mock_dataset_all_valid_genes, **fe_config)

    return weighted_sampling_fe


@fixture
def sorting_truncate_fc(mock_dataset_all_valid_genes, identity_fg_all_valid_genes, identity_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 2,
            "num_metadata_tokens": 0,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
        },
    )

    identity_fg_all_valid_genes.preprocess_embeddings()
    identity_fe.preprocess_embeddings()

    sorting_truncate_fc = SortingTruncateFc(
            identity_fg_all_valid_genes,
            identity_fe,
            mock_dataset_all_valid_genes,
            **fc_config,
    )

    metadata_embeddings = instantiate_from_config(sorting_truncate_fc.embedding_parameters)

    return sorting_truncate_fc




@fixture
def sorting_random_sample_fc(mock_dataset_all_valid_genes, identity_fg_all_valid_genes, identity_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 2,
            "num_metadata_tokens": 0,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
        },
    )

    identity_fg_all_valid_genes.preprocess_embeddings()
    identity_fe.preprocess_embeddings()

    sorting_random_sample_fc = SortingRandomSampleFc(
            identity_fg_all_valid_genes,
            identity_fe,
            mock_dataset_all_valid_genes,
            **fc_config,
    )

    metadata_embeddings = instantiate_from_config(sorting_random_sample_fc.embedding_parameters)

    return sorting_random_sample_fc





@fixture
def sorting_weighted_resample_fc(mock_dataset_all_valid_genes, identity_fg_all_valid_genes, identity_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 2,
            "sample_size": 2,
            "num_metadata_tokens": 0,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
        },
    )

    identity_fg_all_valid_genes.preprocess_embeddings()
    identity_fe.preprocess_embeddings()

    sorting_weighted_resample_fc = SortingWeightedResampleFc(
            identity_fg_all_valid_genes,
            identity_fe,
            mock_dataset_all_valid_genes,
            **fc_config,
    )

    metadata_embeddings = instantiate_from_config(sorting_weighted_resample_fc.embedding_parameters)

    return sorting_weighted_resample_fc


@fixture
def geneformer_fc(zero_expression_mock_dataset, zero_expression_identity_fg, zero_expression_sorting_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 4,
            "num_metadata_tokens": 0,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
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

    metadata_embeddings = instantiate_from_config(geneformer_fc.embedding_parameters)

    return geneformer_fc


@fixture
def scgpt_fc(zero_expression_mock_dataset, zero_expression_identity_fg, zero_expression_binning_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 2,
            "num_metadata_tokens": 0,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
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

    metadata_embeddings = instantiate_from_config(scgpt_fc.embedding_parameters)

    return scgpt_fc


@fixture
def chrom_sort_random_sample_fc(chrom_mock_dataset,
                                identity_fg_all_valid_genes,
                                identity_fe):
    if "DATA_PATH" not in os.environ:
        pytest.skip(".env file must specify DATA_PATH for ChromSortRandomSampleFc test.")
    
    adata, meta_path = chrom_mock_dataset


    fc_config = OmegaConf.create(
        {
            "max_input_length": 6,   # make it smaller than full chrom seq so limit() runs
            "num_metadata_tokens": 0,
            "gene_metadata_filepath": str(meta_path),
            "ensembl_dir": os.environ["DATA_PATH"],
            "species": "human",
            "embedding_parameters": { "type": "torch.nn.Module" },
        }
    )

    # FG only (no FE needed)
    identity_fg_all_valid_genes.preprocess_embeddings()
    identity_fe.preprocess_embeddings()

    fc = ChromSortRandomSampleFc(
        fg=identity_fg_all_valid_genes,
        fe=identity_fe,
        adata=adata,
        **fc_config,
    )

    # deterministic sampling
    fc.rng = np.random.default_rng(0)

    return fc



@fixture
def uce_fc(mock_dataset_all_valid_genes, identity_fg_all_valid_genes, weighted_sampling_fe):
    if "DATA_PATH" not in os.environ:
        pytest.skip(".env file must specify DATA_PATH for UCE `Fc` test.")

    fc_config = OmegaConf.create(
        {
            "max_input_length": 4,
            "num_metadata_tokens": 100,
            "ensembl_dir": os.environ["DATA_PATH"],
            "species": "human",
            "gene_metadata_filepath": f"{os.environ['DATA_PATH']}/gene_metadata/species_chrom.csv",
            "embedding_parameters": {
                "type": "torch.nn.Module",
                "type": "Heimdall.embedding.GaussianInitEmbedding",
                "args": {
                    "num_embeddings": 50,
                    "embedding_dim": 128,
                },
            },
        },
    )
    identity_fg_all_valid_genes.preprocess_embeddings()

    valid_mask = mock_dataset_all_valid_genes.var["identity_valid_mask"]
    mock_dataset_all_valid_genes.raw = mock_dataset_all_valid_genes.copy()
    mock_dataset_all_valid_genes = mock_dataset_all_valid_genes[:, valid_mask].copy()

    weighted_sampling_fe.preprocess_embeddings()

    uce_fc = UCEFc(
        identity_fg_all_valid_genes,
        weighted_sampling_fe,
        mock_dataset_all_valid_genes,
        **fc_config,
    )

    metadata_embeddings = instantiate_from_config(uce_fc.embedding_parameters)

    return uce_fc
