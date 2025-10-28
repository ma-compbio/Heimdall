import os
from textwrap import dedent, indent

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from omegaconf import OmegaConf, open_dict
from pytest import fixture

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import setup_experiment
from Heimdall.utils import INPUT_KEYS, get_dtype, instantiate_from_config

load_dotenv()

model_configs = ["transformer", "expression_weighted_sum", "average"]
fc_configs = ["geneformer", "geneformer", "geneformer"]


@fixture(scope="module")
def paired_task_config(request, toy_paired_data_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "model=transformer",
                f"model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                f"dataset=test",
                f"tasks=paired_test",
                f"tasks.args.reducer_config.type={request.param}",
                f"dataset.preprocess_args.data_path={toy_paired_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "fg=identity",
                "fe=2nn",
                "fc=test",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def single_task_config(request, toy_single_data_path):
    model_config, fc_config = request.param
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"model={model_config}",
                f"model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                f"dataset=test",
                f"tasks=test",
                f"dataset.preprocess_args.data_path={toy_single_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "fg=identity",
                "fe=zero",
                f"fc={fc_config}",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def multitask_config(request, toy_single_data_path):
    model_config, fc_config = request.param
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"model={model_config}",
                f"model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                f"dataset=test",
                f"tasks=tasklist_placeholder",
                f"+tasks@tasks.args.subtask_configs.1=test",
                f"+tasks@tasks.args.subtask_configs.2=test",
                f"dataset.preprocess_args.data_path={toy_single_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "fg=identity",
                "fe=zero",
                f"fc={fc_config}",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def expression_only_config(request, toy_single_data_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"model=expression_only",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                f"dataset=test",
                f"tasks=test",
                f"dataset.preprocess_args.data_path={toy_single_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "fg=dummy",
                "fe=dummy",
                f"fc=dummy",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def partition_config(request, toy_partitioned_data_path):
    model_config, fc_config = request.param
    model_config, fc_config = request.param
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"model={model_config}",
                f"model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                f"dataset=test",
                f"tasks=test",
                "tasks.args.dataset_config.type=Heimdall.datasets.PartitionedDataset",
                "tasks.cell_rep_config.type=Heimdall.cell_representations.PartitionedCellRepresentation",
                f"dataset.preprocess_args.data_path={toy_partitioned_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "fg=identity",
                "fe=zero",
                f"fc={fc_config}",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


# @fixture(scope="module")
# def accelerator(config):
#     setup
def instantiate_and_run_model(config):
    experiment_primitives = setup_experiment(config, cpu=False)

    if experiment_primitives is None:
        return

    _, cr, model, _ = experiment_primitives

    # Test execution
    batch = next(iter(cr.dataloaders["train"]))
    inputs = {input_key: batch[input_key] for input_key in INPUT_KEYS if input_key in batch}
    model(inputs=inputs)


@pytest.mark.parametrize(
    "single_task_config",
    zip(model_configs, fc_configs),
    indirect=True,
)
def test_single_task_model_instantiation(single_task_config):
    instantiate_and_run_model(single_task_config)


@pytest.mark.parametrize(
    "multitask_config",
    zip(model_configs, fc_configs),
    indirect=True,
)
def test_multitask_model_instantiation(multitask_config):
    instantiate_and_run_model(multitask_config)


def test_expression_only_instantiation(expression_only_config):
    instantiate_and_run_model(expression_only_config)


@pytest.mark.parametrize(
    "paired_task_config",
    [
        "Heimdall.models.SumReducer",
        "Heimdall.models.MeanReducer",
        "Heimdall.models.SymmetricConcatReducer",
        "Heimdall.models.AsymmetricConcatReducer",
    ],
    indirect=True,
)
def test_paired_task_model_instantiation(paired_task_config):
    instantiate_and_run_model(paired_task_config)


@pytest.mark.parametrize(
    "partition_config",
    zip(model_configs, fc_configs),
    indirect=True,
)
def test_partitioned_model_instantiation(partition_config):
    instantiate_and_run_model(partition_config)
