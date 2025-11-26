from Heimdall.models import HeimdallModel
from Heimdall.utils import count_parameters

# from Heimdall.utils import get_dtype


def setup_model(config, cr, is_main_process=True):
    """Set up Heimdall experiment based on config, including cr, model and
    trainer."""

    # Create the model and the types of inputs that it may use
    # `type` can either be `learned`, which is integer tokens and learned nn.embeddings,
    # or `predefined`, which expects the dataset to prepare batchsize x length x hidden_dim
    # float_dtype = get_dtype(config.float_dtype)

    model = HeimdallModel(
        data=cr,
        model_config=config.model,
    )
    # .to(float_dtype)

    if is_main_process:
        num_params = count_parameters(model)
        print(f"\nModel constructed:\n{model}\nNumber of trainable parameters {num_params:,}\n")

    return model
