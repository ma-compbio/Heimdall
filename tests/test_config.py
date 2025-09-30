from hydra import compose, initialize
from omegaconf import OmegaConf


def test_config_with_overrides():
    with initialize(version_base=None, config_path="../Heimdall/config", job_name="test"):
        cfg = compose(config_name="config", overrides=["fg=identity", "fe=binning_scgpt"])
        resolved_config = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    # assert resolved_config.fe.args.embedding_parameters.args.out_features == resolved_config.fe.args.d_embedding
