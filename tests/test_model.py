import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from omegaconf import OmegaConf
from pytest import fixture

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel


@fixture(scope="module")
def plain_toy_data():
    return ad.AnnData(
        X=np.arange(3 * 5).reshape(5, 3),
        var=pd.DataFrame(index=["ENSG00000142611", "ENSG00000157911", "ENSG00000274917"]),
    )


@fixture(scope="module")
def toy_paried_data_path(pytestconfig, plain_toy_data):
    data_path = pytestconfig.cache.mkdir("toy_data")

    adata = plain_toy_data.copy()
    zeros = sp.csr_matrix((adata.shape[0], adata.shape[0]))
    for i, key in enumerate(("train", "val", "test", "task")):
        adata.obsp[key] = zeros.copy()
        if key != "task":
            adata.obsp[key][i, i] = 1

    path = data_path / "toy_single_adata.h5ad"
    adata.write_h5ad(path)

    return path


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
def paired_task_config(request, toy_paried_data_path):
    config_string = f"""
    project_name: Cell_Cell_interaction
    run_name: run_name
    work_dir: work_dir
    seed: 42
    data_path: null
    ensembl_dir: null
    cache_preprocessed_dataset_dir: null
    entity: Heimdall
    model:
      type: transformer
      args:
        d_model: 128
        pos_enc: BERT
        num_encoder_layers: 2
        nhead: 2
        hidden_act: gelu
        hidden_dropout_prob: 0.1
        attention_probs_dropout_prob: 0.1
        use_flash_attn: false
        pooling: cls_pooling
    dataset:
      dataset_name: zeng_merfish_ccc_subset
      preprocess_args:
        data_path: {toy_paried_data_path}
        top_n_genes: 1000
        normalize: true
        log_1p: true
        scale_data: true
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
      type: Heimdall.fc.GeneformerFc
      args:
        max_input_length: 2048
    fe:
      type: Heimdall.fe.SortingFe
      args:
        embedding_parameters:
          type: torch.nn.Embedding
          args:
            num_embeddings: "max_seq_length"
            embedding_dim: 128
        d_embedding: 128
    fg:
      name: IdentityFg
      type: Heimdall.fg.IdentityFg
      args:
        embedding_parameters:
          type: torch.nn.Embedding
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
    data_path: null
    ensembl_dir: null
    cache_preprocessed_dataset_dir: null
    entity: Heimdall
    model:
      type: transformer
      args:
        d_model: 128
        pos_enc: BERT
        num_encoder_layers: 2
        nhead: 2
        hidden_act: gelu
        hidden_dropout_prob: 0.1
        attention_probs_dropout_prob: 0.1
        use_flash_attn: false
        pooling: cls_pooling
    dataset:
      dataset_name: cell_type_classification
      preprocess_args:
        data_path: {toy_single_data_path}
        top_n_genes: 1000
        normalize: true
        log_1p: true
        scale_data: true
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
      type: Heimdall.fc.ScGPTFc
      args:
        max_input_length: 2048
    fe:
      type: Heimdall.fe.BinningFe
      args:
        d_embedding: 128
        num_bins: 10
        embedding_parameters:
          type: Heimdall.utils.FlexibleTypeLinear
          args:
            in_features: "max_seq_length"
            out_features: 128
    fg:
      name: IdentityFg
      type: Heimdall.fg.IdentityFg
      args:
        embedding_parameters:
          type: torch.nn.Embedding
          args:
            num_embeddings: "vocab_size"
            embedding_dim: 128
        d_embedding: 128
    loss:
      name: CrossEntropyLoss
    """
    conf = OmegaConf.create(config_string)

    return conf


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
    cr = CellRepresentation(paired_task_config)  # takes in the whole paired_task_config from hydra

    model = HeimdallModel(
        data=cr,
        model_config=paired_task_config.model.args,
        task_config=paired_task_config.tasks.args,
    )

    # Test execution
    batch = next(iter(cr.dataloaders["train"]))
    model(inputs=(batch["identity_inputs"], batch["expression_inputs"]), conditional_tokens=None)


def test_single_task_model_instantiation(single_task_config):
    cr = CellRepresentation(single_task_config)  # takes in the whole config from hydra

    model = HeimdallModel(
        data=cr,
        model_config=single_task_config.model.args,
        task_config=single_task_config.tasks.args,
    )

    # Test execution
    batch = next(iter(cr.dataloaders["train"]))
    model(inputs=(batch["identity_inputs"], batch["expression_inputs"]), conditional_tokens=None)
