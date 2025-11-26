import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import hydra
import pytest
from dotenv import load_dotenv

from Heimdall.trainer import PrecomputationContext, setup_trainer

load_dotenv()

if "HYDRA_USER" not in os.environ:
    pytest.skip(".env file must specify HYDRA_USER for integrated test.", allow_module_level=True)


@pytest.mark.integration
def test_default_hydra_train(session_cache_dir):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                # "+experiments_dev=classification_experiment_dev",
                "+experiments=spatial_cancer_split1",
                # "user=lane-nick"
                "model=transformer_small",
                "model.args.use_flash_attn=true",  # Also tests flash-attn ;)
                "fg=pca_esm2",
                "fe=identity",
                "fc=uce",
                "seed=55",
                "project_name=demo",
                "run_wandb=false",
                "tasks.args.epochs=1",
                "fc.args.max_input_length=256",
                "fc.args.tailor_config.args.sample_size=200",
                "work_dir=null",
                f"cache_preprocessed_dataset_dir={session_cache_dir}",
                # f"user={os.environ['HYDRA_USER']}"
            ],
        )
    trainer = setup_trainer(config, cpu=False)
    trainer.fit(resume_from_checkpoint=False, checkpoint_every_n_epochs=20)

    valid_log, _ = trainer.validate_model(trainer.dataloader_val, dataset_type="valid")

    for subtask_name, subtask in trainer.data.tasklist:
        assert valid_log[f"valid_{subtask_name}_MatthewsCorrCoef"] > 0.25


@pytest.mark.integration
def test_partitioned_hydra_train(session_cache_dir):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                # "+experiments_dev=classification_experiment_dev",
                "+experiments=pretraining",
                "dataset=pretrain_dev",
                # "user=lane-nick"
                "model=transformer_small",
                "seed=55",
                "project_name=demo",
                "run_wandb=false",
                "tasks.args.epochs=1",
                "work_dir=null",
                f"cache_preprocessed_dataset_dir={session_cache_dir}",
                "fc.args.max_input_length=256",
                # f"user={os.environ['HYDRA_USER']}"
                "trainer.args.skip_umaps=true",
            ],
        )

    trainer = setup_trainer(config, cpu=False)
    trainer.fit(resume_from_checkpoint=False, checkpoint_every_n_epochs=20)

    valid_log, _ = trainer.validate_model(trainer.dataloader_val, dataset_type="valid")

    for subtask_name, subtask in trainer.data.tasklist:
        print(valid_log)


@pytest.mark.integration
def test_partitioned_precomputation(session_cache_dir):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                # "+experiments_dev=classification_experiment_dev",
                "+experiments=pretraining",
                "dataset=pretrain_dev",
                # "user=lane-nick"
                "model=transformer_small",
                "seed=55",
                "project_name=demo",
                "run_wandb=false",
                "tasks.args.epochs=1",
                "work_dir=null",
                f"cache_preprocessed_dataset_dir={session_cache_dir}",
                "fc.args.max_input_length=256",
                # f"user={os.environ['HYDRA_USER']}"
                "trainer.args.skip_umaps=true",
            ],
        )

    trainer = setup_trainer(config, cpu=False)

    trainer.data.partition = 0
    print("> Precomputing embeddings...")
    with PrecomputationContext(trainer, save_precomputed=True, get_precomputed=False, run_wandb=False):
        _, full_embed = trainer.validate_model(trainer.dataloader_full, "full")

    print("> Retrieving precomputed embeddings...")
    with PrecomputationContext(trainer, save_precomputed=False, get_precomputed=True, run_wandb=False):
        _, full_embed = trainer.validate_model(trainer.dataloader_full, "full")
