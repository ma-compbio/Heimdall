import hydra

# from datasets import Dataset
from omegaconf import OmegaConf

# Cell representation tools from heimdall
from Heimdall.cell_representations import CellRepresentation

# initialize the model
from Heimdall.models import HeimdallTransformer, TransformerConfig
from Heimdall.trainer import HeimdallTrainer
from Heimdall.utils import count_parameters


# using @hydra.main so that we can take in command line arguments
@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):
    print(OmegaConf.to_yaml(config))

    #####
    # After preparing your f_g and f_c, use the Heimdall Cell_Representation object to load in and
    # preprocess the dataset
    #####

    cr = CellRepresentation(config)  # takes in the whole config from hydra

    ########
    # Create the model and the types of inputs that it may use
    # `type` can either be `learned`, which is integer tokens and learned nn.embeddings,
    # or `predefined`, which expects the dataset to prepare batchsize x length x hidden_dim
    #######

    # TODO: parse from config?
    # conditional_input_types = {
    #     "conditional_tokens_1": {
    #         "type": "learned",
    #         "vocab_size": 1000,
    #     },
    #     "conditional_tokens_2": {
    #         "type": "learned",
    #         "vocab_size": 1000,
    #     },
    # }
    conditional_input_types = None

    # model config based on your specifications
    transformer_config = TransformerConfig(
        vocab_size=cr.sequence_length,
        max_seq_length=cr.sequence_length,
        prediction_dim=cr.num_tasks,
        d_model=config.model.args.hidden_size,
        nhead=config.model.args.num_attention_heads,
        num_encoder_layers=config.model.args.num_hidden_layers,
    )

    model = HeimdallTransformer(
        config=transformer_config,
        input_type="learned",
        conditional_input_types=conditional_input_types,
    )

    num_params = count_parameters(model)

    print(f"{model}")
    print(f"Number of Parameters {num_params}")

    #####
    # Initialize the Trainer
    #####
    trainer = HeimdallTrainer(
        cfg=config,
        model=model,
        dataloader_train=cr.dataloaders["train"],
        dataloader_val=cr.dataloaders["val"],
        dataloader_test=cr.dataloaders["test"],
        run_wandb=True,
    )

    # Training
    trainer.fit()


if __name__ == "__main__":
    main()
