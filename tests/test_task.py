import hydra
import pytest
from omegaconf import OmegaConf

from Heimdall.task import PairedInstanceTask, SingleInstanceTask, Tasklist


def test_instantiate_single_instance_task():
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/tasks"):
        conf = hydra.compose(
            config_name="spatial_cancer_split",
        )
        OmegaConf.resolve(conf)

    data = None
    task = SingleInstanceTask(data, **conf.args)


def test_instantiate_paired_instance_task():
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/tasks"):
        conf = hydra.compose(
            config_name="reverse_perturbation",
        )
        OmegaConf.resolve(conf)

    data = None
    task = PairedInstanceTask(data, **conf.args)


def test_instantiate_tasklist():
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"tasks=tasklist_placeholder",
                "+tasks@tasks.args.subtask_configs.1=spatial_cancer_split",
            ],
        )
        OmegaConf.resolve(conf)

    data = None
    tasklist = Tasklist(data, **conf.tasks.args)
    for _, subtask in tasklist:
        print(f"{subtask=}")


def test_instantiate_multitask_tasklist():
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"tasks=tasklist_placeholder",
                "+tasks@tasks.args.subtask_configs.1=spatial_cancer_split",
                "+tasks@tasks.args.subtask_configs.2=spatial_cancer_split",
            ],
        )
        OmegaConf.resolve(conf)

    data = None
    tasklist = Tasklist(data, **conf.tasks.args)
    for _, subtask in tasklist:
        print(f"{subtask=}")


def test_invalid_tasklist():
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"tasks=tasklist_placeholder",
                "+tasks@tasks.args.subtask_configs.1=spatial_cancer_split",
                "+tasks@tasks.args.subtask_configs.2=new_sctab_split",
            ],
        )
        OmegaConf.resolve(conf)

    data = None
    with pytest.raises(ValueError):
        tasklist = Tasklist(data, **conf.tasks.args)
