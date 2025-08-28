import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytest import fixture

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel
from Heimdall.utils import get_dtype

load_dotenv()


@fixture(scope="module")
def paired_task_config(request, toy_paried_data_path):
    config_string = f"""
    project_name: Cell_Cell_interaction
    run_name: run_name
    work_dir: work_dir
    seed: 42
    float_dtype: 'float32'
    data_path: {os.environ['DATA_PATH']}
    ensembl_dir: {os.environ['DATA_PATH']}
    cache_preprocessed_dataset_dir: null
    entity: Heimdall
    model:
      type: Heimdall.models.HeimdallTransformer
      name: transformer
      args:
        d_model: 128
        pos_enc: BERT
        num_encoder_layers: 6
        nhead: 4
        hidden_act: gelu
        hidden_dropout_prob: 0.1
        attention_probs_dropout_prob: 0.1
        use_flash_attn: false
        pooling: cls_pooling # or "mean_pooling"
    dataset:
      dataset_name: zeng_merfish_ccc_subset
      preprocess_args:
        data_path: {toy_paried_data_path}
        top_n_genes: 1000
        normalize: true
        log_1p: true
        scale_data: false
        species: mouse
    tasks:
      args:
        task_type: binary
        interaction_type: _all_
        label_col_name: class
        splits:
          type: predefined
          keys_:
            train: train
            val: val
            test: test
        metrics:
        - Accuracy
        shuffle: true
        batchsize: 32
        epochs: 10
        prediction_dim: 14
        reduction:
          type: {request.param}
        dataset_config:
          type: Heimdall.datasets.PairedInstanceDataset
        head_config:
          type: Heimdall.models.LinearCellPredHead
          args: null
    scheduler:
      name: cosine
      lr_schedule_type: cosine
      warmup_ratio: 0.1
      num_epochs: 20
    trainer:
      accelerator: cuda
      precision: 32-true
      random_seed: 42
      per_device_batch_size: 64
      accumulate_grad_batches: 1
      grad_norm_clip: 1.0
      fastdev: false
    optimizer:
      name: AdamW
      args:
        lr: 0.0001
        weight_decay: 0.1
        betas:
        - 0.9
        - 0.95
        foreach: false
    fc:
      type: Heimdall.fc.Fc  # Geneformer
      args:
        max_input_length: 2048
        embedding_parameters:
          type: torch.nn.Module  # Should throw an error if called
        tailor_config:
          type: Heimdall.tailor.ReorderTailor
        order_config:
          type: Heimdall.order.ExpressionOrder
        reduce_config:
          type: Heimdall.reduce.IdentityReduce
    fe:
      type: Heimdall.fe.IdentityFe
      args:
        embedding_parameters:
          type: Heimdall.embedding.FlexibleTypeEmbedding
          args:
            num_embeddings: "max_seq_length"
            embedding_dim: 128
        d_embedding: 128
    fg:
      name: IdentityFg
      type: Heimdall.fg.IdentityFg
      args:
        embedding_parameters:
          type: Heimdall.embedding.FlexibleTypeEmbedding
          args:
            num_embeddings: "vocab_size"
            embedding_dim: 128
        d_embedding: 128
    loss:
      name: CrossEntropyLoss
    """
    conf = OmegaConf.create(config_string)

    return conf


@fixture(scope="module")
def single_task_config(toy_single_data_path):
    config_string = f"""
    project_name: Cell_Type_Classification_dev
    run_name: run_name
    work_dir: work_dir
    seed: 42
    float_dtype: 'float32'
    data_path: {os.environ['DATA_PATH']}
    ensembl_dir: {os.environ['DATA_PATH']}
    cache_preprocessed_dataset_dir: null
    entity: Heimdall
    model:
      type: Heimdall.models.ExpressionOnly
      name: logistic_regression
    dataset:
      dataset_name: cell_type_classification
      preprocess_args:
        data_path: {toy_single_data_path}
        top_n_genes: 1000
        normalize: true
        log_1p: true
        scale_data: false
        species: mouse
    tasks:
      args:
        task_type: multiclass
        label_col_name: class
        metrics:
        - Accuracy
        - MatthewsCorrCoef
        train_split: 0.8
        shuffle: true
        batchsize: 32
        epochs: 10
        prediction_dim: 14
        dataset_config:
          type: Heimdall.datasets.SingleInstanceDataset
        head_config:
          type: Heimdall.models.ExpressionOnlyCellPredHead
          args: null
    scheduler:
      name: cosine
      lr_schedule_type: cosine
      warmup_ratio: 0.1
      num_epochs: 20
    trainer:
      accelerator: cuda
      precision: 32-true
      random_seed: 42
      per_device_batch_size: 64
      accumulate_grad_batches: 1
      grad_norm_clip: 1.0
      fastdev: false
    optimizer:
      name: AdamW
      args:
        lr: 0.0001
        weight_decay: 0.1
        betas:
        - 0.9
        - 0.95
        foreach: false
    fc:
      type: Heimdall.fc.DummyFc
      args:
        embedding_parameters:
          type: torch.nn.Module  # Should throw an error if called
        tailor_config:
          type: Heimdall.tailor.Tailor  # Should throw an error if called
        order_config:
          type: Heimdall.order.Order  # Should throw an error if called
        reduce_config:
          type: Heimdall.reduce.Reduce  # Should throw an error if called
    fe:
      type: Heimdall.fe.IdentityFe
      args:
        embedding_parameters:
          type: torch.nn.Module
        d_embedding: null
        drop_zeros: False
    fg:
      name: IdentityFg
      type: Heimdall.fg.IdentityFg
      args:
        embedding_parameters:
          type: torch.nn.module
        d_embedding: null
    loss:
      name: CrossEntropyLoss
    """
    conf = OmegaConf.create(config_string)

    return conf


def instantiate_and_run_model(config):
    cr = CellRepresentation(config)  # takes in the whole config from hydra

    float_dtype = get_dtype(config.float_dtype)

    model = HeimdallModel(
        data=cr,
        model_config=config.model,
        task_config=config.tasks.args,
    ).to(float_dtype)

    # Test execution
    batch = next(iter(cr.dataloaders["train"]))
    model(
        inputs=(batch["identity_inputs"], batch["expression_inputs"]),
        attention_mask=batch["expression_padding"],
    )


def test_single_task_model_instantiation(single_task_config):
    instantiate_and_run_model(single_task_config)


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
