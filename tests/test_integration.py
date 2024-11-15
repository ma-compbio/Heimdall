import os

import hydra
import pytest
from dotenv import load_dotenv
from omegaconf import OmegaConf

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel
from Heimdall.trainer import HeimdallTrainer
from Heimdall.utils import count_parameters, get_dtype

load_dotenv()

if "HYDRA_USER" not in os.environ:
    pytest.skip(".env file must specify HYDRA_USER for integrated test.", allow_module_level=True)


@pytest.mark.integration
def test_default_hydra_train():
    with hydra.initialize(version_base=None, config_path="../config"):
        config = hydra.compose(
            config_name="config",
            overrides=["+experiments=cta_pancreas", f"user={os.environ['HYDRA_USER']}"],
        )
        print(OmegaConf.to_yaml(config))

    cr = CellRepresentation(config)  # takes in the whole config from hydra

    conditional_input_types = None

    float_dtype = get_dtype(config.float_dtype)

    model = HeimdallModel(
        data=cr,
        model_config=config.model,
        task_config=config.tasks.args,
        conditional_input_types=conditional_input_types,
    ).to(float_dtype)

    num_params = count_parameters(model)

    print(f"\nModel constructed:\n{model}\nNumber of trainable parameters {num_params:,}\n")

    trainer = HeimdallTrainer(cfg=config, model=model, data=cr, run_wandb=False)

    trainer.fit()

    valid_log = trainer.validate_model(trainer.dataloader_val, dataset_type="valid")

    assert valid_log["valid_MatthewsCorrCoef"] > 0.8
