import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytest import fixture
from scipy.sparse import csr_array

from Heimdall.cell_representations import CellRepresentation
from Heimdall.fc import ChromosomeAwareFc, Fc
from Heimdall.fe import BinningFe, Fe, IdentityFe, ScBERTBinningFe, ZeroFe
from Heimdall.fg import Fg, IdentityFg
from Heimdall.utils import convert_to_ensembl_ids, instantiate_from_config

load_dotenv()


from dataclasses import dataclass


@dataclass
class MockCellRepresentation(CellRepresentation):
    adata: ad.AnnData
    _cfg: OmegaConf
    verbose: bool = True

    def set_representation_functions(
        self,
        fg: Fg | None = None,
        fe: Fe | None = None,
        fc: Fc | None = None,
    ):
        self.fg = fg
        self.fe = fe
        self.fc = fc


@fixture(scope="module")
def gene_names():
    return ["A1BG", "A1CF", "fake_gene", "A2M"]


@fixture(scope="module")
def valid_gene_names():
    return ["A1BG", "A1CF", "A2M"]


@fixture(scope="module")
def plain_toy_data(valid_gene_names):
    adata = ad.AnnData(
        X=csr_array(np.arange(3 * 10).reshape(10, 3).astype(np.float32)),
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
def toy_paired_data_path(pytestconfig, plain_toy_data):
    data_path = pytestconfig.cache.mkdir("toy_data")

    adata = plain_toy_data.copy()
    zeros = csr_array((adata.shape[0], adata.shape[0]))
    for i, key in enumerate(("train", "val", "test", "task")):
        adata.obsp[key] = zeros.copy()
        if key != "task":
            adata.obsp[key][i, i] = 1

    path = data_path / "toy_paired_adata.h5ad"
    adata.write_h5ad(path)

    return path


@fixture(scope="module")
def toy_partitioned_data_path(pytestconfig, plain_toy_data):
    data_path = pytestconfig.cache.mkdir("toy_data")

    num_partitions = 5
    partition_directory = data_path / "toy_partitioned_adata"
    partition_directory.mkdir(exist_ok=True)
    for partition in range(num_partitions):
        adata = plain_toy_data.copy()
        adata.obs["split"] = "train"
        adata.obs["class"] = 0

        path = partition_directory / f"{partition}.h5ad"
        adata.write_h5ad(path)

    return partition_directory


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

    return MockCellRepresentation(
        adata=mock_dataset,
        _cfg=None,
    )


# @fixture
# def mock_cr(mock_dataset):
#     return MockCellRepresentation(
#         adata=mock_dataset,
#         _cfg=None
#     )


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

    return MockCellRepresentation(
        adata=mock_dataset,
        _cfg=None,
    )


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

    return MockCellRepresentation(
        adata=mock_dataset,
        _cfg=None,
    )


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
def scbert_binning_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": int(np.max(mock_dataset.adata.X)) + 2,
            "embedding_parameters": {
                "type": "Heimdall.embedding.FlexibleTypeLinear",
                "args": {
                    "in_features": 1,  # Replace later
                    "out_features": 128,
                },
            },
            "d_embedding": 128,
            "num_bins": 2,
        },
    )
    binning_fe = ScBERTBinningFe(mock_dataset, **fe_config)

    return binning_fe


@fixture
def binning_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": int(np.max(mock_dataset.adata.X)) + 2,
            "embedding_parameters": {
                "type": "Heimdall.embedding.FlexibleTypeLinear",
                "args": {
                    "in_features": 1,  # Replace later
                    "out_features": 128,
                },
            },
            "d_embedding": 128,
            "num_bins": int(np.max(mock_dataset.adata.X)),
        },
    )
    binning_fe = BinningFe(mock_dataset, **fe_config)

    return binning_fe


@fixture
def identity_fe(mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": int(np.max(mock_dataset.adata.X)) + 3,
            "embedding_parameters": {
                "type": "Heimdall.embedding.TwoLayerNN",
                "args": {
                    "in_features": 1,
                    "out_features": 128,
                },
            },
            "d_embedding": 128,
        },
    )
    identity_fe = IdentityFe(mock_dataset, **fe_config)

    return identity_fe


@fixture
def zero_expression_identity_fe(zero_expression_mock_dataset):
    fe_config = OmegaConf.create(
        {
            "vocab_size": int(np.max(zero_expression_mock_dataset.adata.X)) + 3,
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "args": {
                    "num_embeddings": "vocab_size",
                    "embedding_dim": 128,
                },
            },
            "d_embedding": 128,
        },
    )
    identity_fe = IdentityFe(zero_expression_mock_dataset, **fe_config)

    return identity_fe


@fixture
def identity_fe_all_valid_genes(mock_dataset_all_valid_genes):
    fe_config = OmegaConf.create(
        {
            "vocab_size": int(np.max(mock_dataset_all_valid_genes.adata.X)) + 3,
            "embedding_parameters": {
                "type": "torch.nn.Embedding",
                "args": {
                    "num_embeddings": "vocab_size",
                    "embedding_dim": 128,
                },
            },
            "d_embedding": 128,
        },
    )
    identity_fe = IdentityFe(mock_dataset_all_valid_genes, **fe_config)

    return identity_fe


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
            "num_bins": int(np.max(zero_expression_mock_dataset.adata.X)),
        },
    )
    binning_fe = BinningFe(zero_expression_mock_dataset, **fe_config)

    return binning_fe


@fixture
def zero_expression_scbert_binning_fe(zero_expression_mock_dataset):
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
            "num_bins": 2,
        },
    )
    binning_fe = ScBERTBinningFe(zero_expression_mock_dataset, **fe_config)

    return binning_fe


@fixture
def geneformer_fc(zero_expression_mock_dataset, zero_expression_identity_fg, zero_expression_identity_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 4,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
            "tailor_config": {
                "type": "Heimdall.tailor.ReorderTailor",
            },
            "order_config": {
                "type": "Heimdall.order.ExpressionOrder",
            },
            "reduce_config": {
                "type": "Heimdall.reduce.IdentityReduce",
            },
        },
    )
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_identity_fe.preprocess_embeddings()

    geneformer_fc = Fc(
        zero_expression_identity_fg,
        zero_expression_identity_fe,
        zero_expression_mock_dataset,
        **fc_config,
    )

    metadata_embeddings = instantiate_from_config(geneformer_fc.embedding_parameters)

    return geneformer_fc


@fixture
def scgpt_fc(zero_expression_mock_dataset, zero_expression_identity_fg, zero_expression_binning_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 3,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
            "tailor_config": {
                "type": "Heimdall.tailor.ReorderTailor",
            },
            "order_config": {
                "type": "Heimdall.order.RandomOrder",
            },
            "reduce_config": {
                "type": "Heimdall.reduce.SumReduce",
            },
        },
    )
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_binning_fe.preprocess_embeddings()

    scgpt_fc = Fc(
        zero_expression_identity_fg,
        zero_expression_binning_fe,
        zero_expression_mock_dataset,
        **fc_config,
    )

    metadata_embeddings = instantiate_from_config(scgpt_fc.embedding_parameters)

    return scgpt_fc


@fixture
def scbert_fc(zero_expression_mock_dataset, zero_expression_identity_fg, zero_expression_scbert_binning_fe):
    fc_config = OmegaConf.create(
        {
            "max_input_length": 3,
            "embedding_parameters": {
                "type": "torch.nn.Module",
            },
            "tailor_config": {
                "type": "Heimdall.tailor.ReorderTailor",
            },
            "order_config": {
                "type": "Heimdall.order.RandomOrder",
            },
            "reduce_config": {
                "type": "Heimdall.reduce.SumReduce",
            },
        },
    )
    zero_expression_identity_fg.preprocess_embeddings()
    zero_expression_scbert_binning_fe.preprocess_embeddings()

    scbert_fc = Fc(
        zero_expression_identity_fg,
        zero_expression_scbert_binning_fe,
        zero_expression_mock_dataset,
        **fc_config,
    )

    metadata_embeddings = instantiate_from_config(scbert_fc.embedding_parameters)

    return scbert_fc


@fixture
def uce_fc(mock_dataset_all_valid_genes, identity_fg_all_valid_genes, identity_fe_all_valid_genes):
    if "DATA_PATH" not in os.environ:
        pytest.skip(".env file must specify DATA_PATH for UCE `Fc` test.")

    fc_config = OmegaConf.create(
        {
            "max_input_length": 4,
            "ensembl_dir": os.environ["DATA_PATH"],
            "species": "human",
            "gene_metadata_filepath": f"{os.environ['DATA_PATH']}/gene_metadata/species_chrom.csv",
            "embedding_parameters": {
                "type": "torch.nn.Module",
                "type": "Heimdall.embedding.GaussianInitEmbedding",
                "args": {
                    "num_embeddings": 100,
                    "embedding_dim": 128,
                },
            },
            "tailor_config": {
                "type": "Heimdall.tailor.ChromosomeTailor",
                "args": {
                    "sample_size": 5,
                },
            },
            "order_config": {
                "type": "Heimdall.order.ChromosomeOrder",
            },
            "reduce_config": {
                "type": "Heimdall.reduce.ChromosomeReduce",
            },
        },
    )
    identity_fg_all_valid_genes.preprocess_embeddings()

    # valid_mask = mock_dataset_all_valid_genes.adata.var["identity_valid_mask"]
    # mock_dataset_all_valid_genes.adata.raw = mock_dataset_all_valid_genes.adata.copy()
    # mock_dataset_all_valid_genes.adata = mock_dataset_all_valid_genes[:, valid_mask].copy()

    identity_fe_all_valid_genes.preprocess_embeddings()

    uce_fc = ChromosomeAwareFc(
        identity_fg_all_valid_genes,
        identity_fe_all_valid_genes,
        mock_dataset_all_valid_genes,
        **fc_config,
    )

    metadata_embeddings = instantiate_from_config(uce_fc.embedding_parameters)

    return uce_fc


@fixture(scope="session")
def session_cache_dir(tmp_path_factory):
    # Create the directory using tmp_path_factory
    cache_dir = tmp_path_factory.mktemp("cache")
    yield cache_dir
