## loading in libraries
import anndata as ad
import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import SchedulerType, get_scheduler

## Cell representation tools from heimdall
from Heimdall.cell_representations import CellRepresentation
from Heimdall.f_c import geneformer_fc
from Heimdall.f_g import identity_fg

## initialize the model
from Heimdall.models import HeimdallTransformer, TransformerConfig
from Heimdall.trainer import HeimdallTrainer
from Heimdall.utils import heimdall_collate_fn


### using @hydra.main so that we can take in command line arguments
@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):
    print(OmegaConf.to_yaml(config))

    #####
    # After preparing your f_g and f_c, use the Heimdall Cell_Representation object to load in and
    # preprocess the dataset
    #####

    CR = CellRepresentation(config)  ## takes in the whole config from hydra
    CR.preprocess_anndata()  ## standard sc preprocessing can be done here
    CR.preprocess_f_g(identity_fg)  ## takes in the identity f_g specified above
    CR.preprocess_f_c(geneformer_fc)  ## takes in the geneformer f_c specified above
    CR.prepare_labels()  ## prepares the labels

    ## we can take this out here now and pass this into a PyTorch dataloader and separately create the model
    X = CR.cell_representation
    y = CR.labels

    print(f"Cell representation X: {X.shape}")
    print(f"Cell labels y: {y.shape}")

    ########
    # PREPARE THE DATASET
    # I am including this explicit example here just for completeness, but this can
    # easily be rolled into a helper function
    ########

    train_x, test_val_x, train_y, test_val_y = train_test_split(X, y, test_size=0.2, random_state=42)
    test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, test_size=0.5, random_state=42)

    print(f"> Cell representation X: {X.shape}")
    print(f"> Cell labels y: {y.shape}")
    print(f"> train_x.shape {train_x.shape}")
    print(f"> validation_x.shape {val_x.shape}")
    print(f"> test_x.shape {test_x.shape}")

    # this is how you dynamically process your outputs into the right dataloader format
    # if you do not want conditional tokens, just omit those arguments
    # what is crucial is that the dataset contains the arguments `inputs` and `labels`, anything else will be put into `conditional`
    ds_train = Dataset.from_dict(
        {"inputs": train_x, "labels": train_y, "conditional_tokens_1": train_x, "conditional_tokens_2": train_x},
    )
    ds_valid = Dataset.from_dict(
        {"inputs": val_x, "labels": val_y, "conditional_tokens_1": val_x, "conditional_tokens_2": val_x},
    )
    ds_test = Dataset.from_dict(
        {"inputs": test_x, "labels": test_y, "conditional_tokens_1": test_x, "conditional_tokens_2": test_x},
    )

    ## this can probably be rolled into the train functionality itself, but lets keep it outside to be eaiser to debug
    dataloader_train = DataLoader(
        ds_train,
        batch_size=int(config.dataset.task_args.batchsize),
        shuffle=config.dataset.task_args.shuffle,
        collate_fn=heimdall_collate_fn,
    )
    dataloader_val = DataLoader(
        ds_valid,
        batch_size=int(config.dataset.task_args.batchsize),
        shuffle=config.dataset.task_args.shuffle,
        collate_fn=heimdall_collate_fn,
    )
    dataloader_test = DataLoader(
        ds_test,
        batch_size=int(config.dataset.task_args.batchsize),
        shuffle=config.dataset.task_args.shuffle,
        collate_fn=heimdall_collate_fn,
    )

    ########
    # Create the model and the types of inputs that it may use
    ## `type` can either be `learned`, which is integer tokens and learned nn.embeddings,
    ##  or `predefined`, which expects the dataset to prepare batchsize x length x hidden_dim
    #######

    conditional_input_types = {
        "conditional_tokens_1": {
            "type": "learned",
            "vocab_size": 1000,
        },
        "conditional_tokens_2": {
            "type": "learned",
            "vocab_size": 1000,
        },
    }

    ## model config based on your specifications
    transformer_config = TransformerConfig(vocab_size=1000, max_seq_length=1000, prediction_dim=20)
    model = HeimdallTransformer(
        config=transformer_config,
        input_type="learned",
        conditional_input_types=conditional_input_types,
    )

    ## optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        foreach=False,
    )  ## the forearch is due to a distributed bug with cosine scheduler

    #####
    # Initialize the Trainer
    #####
    trainer = HeimdallTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataloader_test=dataloader_test,
        run_wandb=True,
    )

    ### Training
    trainer.train()


if __name__ == "__main__":
    main()
