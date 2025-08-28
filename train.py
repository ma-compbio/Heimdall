import hydra
from omegaconf import OmegaConf, open_dict

from Heimdall.models import HeimdallModel
from Heimdall.trainer import HeimdallTrainer
from Heimdall.utils import count_parameters, get_dtype, instantiate_from_config


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))

    # After preparing your f_g and f_c, use the Heimdall Cell_Representation object to load in and
    # preprocess the dataset

    with open_dict(config):
        only_preprocess_data = config.pop("only_preprocess_data", None)
        # pop so hash of cfg is not changed depending on value

    cr = instantiate_from_config(config.tasks.args.cell_rep_config, config)

    if only_preprocess_data:
        return

    # Create the model and the types of inputs that it may use
    # `type` can either be `learned`, which is integer tokens and learned nn.embeddings,
    # or `predefined`, which expects the dataset to prepare batchsize x length x hidden_dim
    float_dtype = get_dtype(config.float_dtype)

    model = HeimdallModel(
        data=cr,
        model_config=config.model,
        task_config=config.tasks.args,
    ).to(float_dtype)

    print(model)

    num_params = count_parameters(model)

    print(f"\nModel constructed:\n{model}\nNumber of trainable parameters {num_params:,}\n")

    trainer = HeimdallTrainer(cfg=config, model=model, data=cr, run_wandb=True)

    trainer.fit()


if __name__ == "__main__":
    main()
