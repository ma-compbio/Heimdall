import hydra
from accelerate import Accelerator
from omegaconf import OmegaConf, open_dict

from Heimdall.models import HeimdallModel
from Heimdall.trainer import HeimdallTrainer
from Heimdall.utils import count_parameters, get_dtype, instantiate_from_config


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):

    # get accelerate context
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = "wandb"
    accelerator_log_kwargs["project_dir"] = config.work_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=config.trainer.accumulate_grad_batches,
        step_scheduler_with_optimizer=False,
        **accelerator_log_kwargs,
    )
    if accelerator.is_main_process:
        print(OmegaConf.to_yaml(config, resolve=True))

    # After preparing your f_g and f_c, use the Heimdall Cell_Representation object to load in and
    # preprocess the dataset

    with open_dict(config):
        only_preprocess_data = config.pop("only_preprocess_data", None)
        # pop so hash of cfg is not changed depending on value

    cr = instantiate_from_config(config.tasks.args.cell_rep_config, config, accelerator)

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

    if accelerator.is_main_process:
        print(model)
        num_params = count_parameters(model)
        print(f"\nModel constructed:\n{model}\nNumber of trainable parameters {num_params:,}\n")

    trainer = HeimdallTrainer(cfg=config, model=model, data=cr, accelerate_context=accelerator, run_wandb=True)

    trainer.fit()


if __name__ == "__main__":
    main()
