import os
import time

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytest import fixture
from torch import nn

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel, setup_experiment
from Heimdall.utils import INPUT_KEYS, get_dtype, instantiate_from_config

load_dotenv()

try:
    from Heimdall.models._flash_attn import FlashTransformerEncoderLayer
except ImportError:
    pytest.skip("`flash_attn` must be installed for `FlashAttentionTransformerEncoder` test.", allow_module_level=True)


@fixture(scope="module")
def flash_attn_config(toy_single_data_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"model=transformer_flash",
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
                f"fc=geneformer",
                "float_dtype=float16",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


def instantiate_and_run_model(config):
    experiment_primitives = setup_experiment(config, cpu=False)

    if experiment_primitives is None:
        return

    _, cr, model, _ = experiment_primitives

    # Test execution
    batch = next(iter(cr.dataloaders["train"]))
    inputs = {input_key: batch[input_key] for input_key in INPUT_KEYS if input_key in batch}
    model(inputs=inputs)


def test_flash_attn_instantiation(flash_attn_config):
    instantiate_and_run_model(flash_attn_config)


# def test_flash_attention_instantiation(flash_attention_config):
#     cr = CellRepresentation(flash_attention_config)  # takes in the whole flash_attention_config from hydra
#
#     float_dtype = get_dtype(flash_attention_config.float_dtype)
#
#     try:
#         model = (
#             HeimdallModel(
#                 data=cr,
#                 model_config=flash_attention_config.model,
#                 task_config=flash_attention_config.tasks.args,
#             )
#             .to(float_dtype)
#             .to("cuda")
#         )
#     except ImportError:
#         pytest.skip(
#             "`flash_attn` must be installed for `FlashAttentionTransformerEncoder` test.",
#             allow_module_level=True,
#         )
#
#     assert model.encoder.use_flash_attn
#
#     # Test execution
#     batch = next(iter(cr.dataloaders["train"]))
#     model(
#         inputs=(batch["identity_inputs"].to("cuda"), batch["expression_inputs"].to("cuda")),
#         attention_mask=batch["expression_padding"].to("cuda"),
#         conditional_tokens=None,
#     )


def test_speed():
    class TestFlashTransformerEncoderLayer(FlashTransformerEncoderLayer):
        def forward(self, input_tensor):
            return super().forward(input_tensor, cu_seqlens=None, max_seqlen=None)

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
    seq_len = 10000  # Long sequence length
    batch_size = 4

    # Initialize models
    torch_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(torch.bfloat16)

    flash_layer = TestFlashTransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
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

    assert (torch_fwd_time / flash_fwd_time) > 1.5
    assert (torch_bwd_time / flash_bwd_time) > 1.1

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
