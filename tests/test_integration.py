import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import hydra
import pytest
from accelerate import Accelerator
from dotenv import load_dotenv
from omegaconf import OmegaConf

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel
from Heimdall.trainer import setup_trainer
from Heimdall.utils import count_parameters, get_dtype, instantiate_from_config

load_dotenv()

if "HYDRA_USER" not in os.environ:
    pytest.skip(".env file must specify HYDRA_USER for integrated test.", allow_module_level=True)


@pytest.mark.integration
def test_default_hydra_train():
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                # "+experiments_dev=classification_experiment_dev",
                "+experiments=spatial_cancer_split1",
                # "user=lane-nick"
                "model=transformer",
                "fg=pca_esm2",
                "fe=identity",
                "fc=uce",
                "seed=55",
                "project_name=demo",
                "run_wandb=false",
                "tasks.args.epochs=1",
                "fc.args.max_input_length=512",
                "fc.args.tailor_config.args.sample_size=450",
                # f"user={os.environ['HYDRA_USER']}"
            ],
        )
    trainer = setup_trainer(config, cpu=False)
    trainer.fit(resume_from_checkpoint=False, checkpoint_every_n_epochs=20)

    valid_log, _ = trainer.validate_model(trainer.dataloader_val, dataset_type="valid")

    for subtask_name, subtask in trainer.data.tasklist:
        assert valid_log[f"valid_{subtask_name}_MatthewsCorrCoef"] > 0.25


@pytest.mark.integration
def test_partitioned_hydra_train():
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                # "+experiments_dev=classification_experiment_dev",
                "+experiments=pretraining",
                "dataset=pretrain_dev",
                # "user=lane-nick"
                "model=transformer",
                "model.args.d_model=128",
                "seed=55",
                "project_name=demo",
                "run_wandb=false",
                "tasks.args.epochs=1",
                "fc.args.max_input_length=512",
                # f"user={os.environ['HYDRA_USER']}"
            ],
        )

    trainer = setup_trainer(config, cpu=False)
    trainer.fit(resume_from_checkpoint=False, checkpoint_every_n_epochs=20)

    valid_log, _ = trainer.validate_model(trainer.dataloader_val, dataset_type="valid")

    for subtask_name, subtask in trainer.data.tasklist:
        assert valid_log[f"valid_{subtask_name}_MatthewsCorrCoef"] > 0
