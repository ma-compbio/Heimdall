import time

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from omegaconf import OmegaConf
from pytest import fixture
from torch import nn

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel
from Heimdall.utils import get_dtype

try:
    from Heimdall.flash_attention_models import FlashAttentionTransformerEncoderLayer
except ImportError:
    pytest.skip("`flash_attn` must be installed for `FlashAttentionTransformerEncoder` test.", allow_module_level=True)


@fixture(scope="module")
def plain_toy_data():
    return ad.AnnData(
        X=np.arange(3 * 5).reshape(5, 3),
        var=pd.DataFrame(index=["ENSG00000142611", "ENSG00000157911", "ENSG00000274917"]),
    )


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
def flash_attention_config(toy_single_data_path):
    config_string = f"""
    project_name: Cell_Type_Classification_dev
    run_name: run_name
    work_dir: work_dir
    seed: 42
    float_dtype: 'float16' # Necessary for FlashAttention
    data_path: null
    ensembl_dir: null
    cache_preprocessed_dataset_dir: null
    entity: Heimdall
    model:
      type: Heimdall.models.HeimdallTransformer
      name: transformer
      args:
        d_model: 128
        pos_enc: BERT
        pooling: cls_pooling
        encoder_layer_parameters:
          type: Heimdall.flash_attention_models.FlashAttentionTransformerEncoderLayer
          args:
            d_model: 128
            nhead: 4
            activation: gelu
            dropout: 0.1
            dim_feedforward: 512
            batch_first: True
            norm_first: True
        encoder_parameters:
          type: Heimdall.flash_attention_models.FlashAttentionTransformerEncoder
          args:
            num_layers: 6
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
      type: Heimdall.fc.GeneformerFc
      args:
        max_input_length: 2048
    fe:
      type: Heimdall.fe.SortingFe
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


def test_flash_attention_instantiation(flash_attention_config):
    cr = CellRepresentation(flash_attention_config)  # takes in the whole flash_attention_config from hydra

    float_dtype = get_dtype(flash_attention_config.float_dtype)

    try:
        model = (
            HeimdallModel(
                data=cr,
                model_config=flash_attention_config.model,
                task_config=flash_attention_config.tasks.args,
            )
            .to(float_dtype)
            .to("cuda")
        )
    except ImportError:
        pytest.skip(
            "`flash_attn` must be installed for `FlashAttentionTransformerEncoder` test.",
            allow_module_level=True,
        )

    # Test execution
    batch = next(iter(cr.dataloaders["train"]))
    model(
        inputs=(batch["identity_inputs"].to("cuda"), batch["expression_inputs"].to("cuda")),
        attention_mask=batch["expression_padding"].to("cuda"),
        conditional_tokens=None,
    )


def test_speed():
    # Speed test function
    def speed_test(model, input_tensor, num_iters=10, backward=False):
        device = input_tensor.device
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Warm-up to ensure consistent timings
        for _ in range(5):
            output = model(input_tensor)
            if backward:
                loss = output.sum()
                loss.backward()
                optimizer.zero_grad()

        # Measure time
        start_time = time.time()
        for _ in range(num_iters):
            output = model(input_tensor)
            if backward:
                loss = output.sum()
                loss.backward()
                optimizer.zero_grad()
        torch.cuda.synchronize(device)  # Ensure all kernels finish
        end_time = time.time()

        elapsed_time = (end_time - start_time) / num_iters
        return elapsed_time

    # Parameters
    d_model = 64
    nhead = 2
    dim_feedforward = 256
    dropout = 0.0
    norm_first = False
    batch_first = False
    seq_len = 10000  # Long sequence length
    batch_size = 4

    # Initialize models
    torch_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
        norm_first=norm_first,
    ).to(torch.bfloat16)

    flash_layer = FlashAttentionTransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        norm_first=norm_first,
        batch_first=batch_first,
    ).to(torch.bfloat16)

    # Generate dummy input
    device = torch.device("cuda")
    dummy_input = torch.rand((seq_len, batch_size, d_model), device=device, dtype=torch.bfloat16)  # (S, N, E)

    # Forward pass speed test
    torch_fwd_time = speed_test(torch_layer, dummy_input, backward=False)
    flash_fwd_time = speed_test(flash_layer, dummy_input, backward=False)

    # Backward pass speed test
    torch_bwd_time = speed_test(torch_layer, dummy_input, backward=True)
    flash_bwd_time = speed_test(flash_layer, dummy_input, backward=True)

    # Print results
    print(f"Forward Pass Time (PyTorch Layer): {torch_fwd_time:.6f} seconds")
    print(f"Forward Pass Time (FlashAttention Layer): {flash_fwd_time:.6f} seconds")
    print(f"Backward Pass Time (PyTorch Layer): {torch_bwd_time:.6f} seconds")
    print(f"Backward Pass Time (FlashAttention Layer): {flash_bwd_time:.6f} seconds")

    assert (torch_fwd_time / flash_fwd_time) > 3
    assert (torch_bwd_time / flash_bwd_time) > 3

    # # Run both models
    # torch_output = torch_layer(dummy_input)
    # flash_output = flash_layer(dummy_input)
    #
    # # Check that outputs are the same
    # try:
    #     assert_allclose(torch_output, flash_output, rtol=1e-5, atol=1e-5)
    #     print("Outputs are identical!")
    # except AssertionError as e:
    #     print("Outputs are different:", e)
