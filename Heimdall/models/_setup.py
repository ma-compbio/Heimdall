from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf, open_dict

from Heimdall.models import HeimdallModel
from Heimdall.utils import count_parameters, get_dtype, instantiate_from_config


def setup_accelerator(config, cpu=False, run_wandb=False):
    # get accelerate context
    accelerator_log_kwargs = {}
    if run_wandb:
        accelerator_log_kwargs["log_with"] = "wandb"
        accelerator_log_kwargs["project_dir"] = config.work_dir

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.trainer.accumulate_grad_batches,
        step_scheduler_with_optimizer=False,
        cpu=cpu,
        mixed_precision="bf16",
        **accelerator_log_kwargs,
        kwargs_handlers=[ddp_kwargs],
    )

    return accelerator


def setup_experiment(config, cpu=False, accelerator=None):
    """Set up Heimdall experiment based on config, including cr, model and
    trainer."""

    run_wandb = getattr(config, "run_wandb", False)
    if accelerator is None:
        accelerator = setup_accelerator(config, cpu=cpu, run_wandb=run_wandb)

    if accelerator.is_main_process:
        print(OmegaConf.to_yaml(config, resolve=True))

    with open_dict(config):
        only_preprocess_data = config.pop("only_preprocess_data", None)
        # pop so hash of cfg is not changed depending on value

    cr = instantiate_from_config(config.tasks.cell_rep_config, config, accelerator)

    if only_preprocess_data:
        return

    # Create the model and the types of inputs that it may use
    # `type` can either be `learned`, which is integer tokens and learned nn.embeddings,
    # or `predefined`, which expects the dataset to prepare batchsize x length x hidden_dim
    float_dtype = get_dtype(config.float_dtype)

    model = HeimdallModel(
        data=cr,
        model_config=config.model,
    )

    model.to(float_dtype)  # to dtype after potentially loading pretrained weights instead of before

    if accelerator.is_main_process:
        num_params = count_parameters(model)
        print(f"\nModel constructed:\n{model}\nNumber of trainable parameters {num_params:,}\n")

    return accelerator, cr, model, run_wandb
