"""Heimdall trainer."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from anndata import AnnData
from omegaconf import OmegaConf
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score, MatthewsCorrCoef, Precision, Recall
from torchmetrics.regression import MeanSquaredError, R2Score
from tqdm import tqdm
from transformers import get_scheduler

import Heimdall.datasets
import Heimdall.losses
import wandb


class HeimdallTrainer:
    def __init__(
        self,
        cfg,
        model,
        data,
        run_wandb=False,
        custom_loss_func=None,
        custom_metrics=None,
    ):
        self.cfg = cfg
        self.model = model
        self.data = data

        # cell type label
        # label_key = self.cfg.tasks.args.label_col_name
        # if not pd.api.types.is_categorical_dtype(self.data.adata.obs[label_key]):
        #    self.data.adata.obs[label_key] = self.data.adata.obs[label_key].astype("category")

        # class_names will now align with integer labels returned by .codes
        # self.class_names = self.data.adata.obs[label_key].cat.categories.tolist()

        # assert len(self.class_names) == self.num_labels, "Mismatch between classes and label indices"

        args = self.cfg.tasks.args

        # Unified label key handling: support .obs or .obsm
        label_key = getattr(args, "label_col_name", None)
        label_obsm_key = getattr(args, "label_obsm_name", None)

        if label_key is not None:
            # Single-label classification using .obs[label_key]
            if not pd.api.types.is_categorical_dtype(self.data.adata.obs[label_key]):
                self.data.adata.obs[label_key] = self.data.adata.obs[label_key].astype("category")
            self.class_names = self.data.adata.obs[label_key].cat.categories.tolist()
            self.num_labels = len(self.class_names)

        elif label_obsm_key is not None:
            # Multi-label classification using .obsm[label_obsm_key]
            self.class_names = self.data.adata.obsm[label_obsm_key].columns.tolist()
            self.num_labels = len(self.class_names)

        else:
            # Auto infering
            self.class_names = data.adata.uns["task_order"]  # NOTE: first entry might be NULL
            self.num_labels = data.num_tasks

        # else:
        #    raise ValueError("Must specify either `label_col_name` or `label_obsm_name` in the config.")

        # Verify model output matches number of labels
        # assert self.num_labels == self.model.output_dim, \
        #     f"Mismatch between number of labels ({self.num_labels}) and model output dim ({self.model.output_dim})"

        self.run_wandb = run_wandb
        self.process = psutil.Process()
        self.custom_loss_func = custom_loss_func
        self.custom_metrics = custom_metrics or {}

        accelerator_log_kwargs = {}
        if run_wandb:
            accelerator_log_kwargs["log_with"] = "wandb"
            accelerator_log_kwargs["project_dir"] = cfg.work_dir
        set_seed(cfg.seed)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.trainer.accumulate_grad_batches,
            step_scheduler_with_optimizer=False,
            **accelerator_log_kwargs,
        )

        if hasattr(model.encoder, "use_flash_attn") and model.encoder.use_flash_attn:
            assert self.accelerator.mixed_precision == "bf16", "If using Flash Attention, mixed precision must be bf16"

        self.optimizer = self._initialize_optimizer()
        self.loss_fn = self._get_loss_function()

        self.accelerator.wait_for_everyone()
        self.print_r0(f"> Using Device: {self.accelerator.device}")
        self.print_r0(f"> Number of Devices: {self.accelerator.num_processes}")

        self._initialize_wandb()
        self._initialize_lr_scheduler()
        self.step = 0

        (
            self.model,
            self.optimizer,
            self.dataloader_train,
            self.dataloader_val,
            self.dataloader_test,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.dataloader_train,
            self.dataloader_val,
            self.dataloader_test,
            self.lr_scheduler,
        )

        if self.accelerator.is_main_process:
            print("> Finished Wrapping the model, optimizer, and dataloaders in accelerate")
            print("> run HeimdallTrainer.train() to begin training")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        for split in ["train", "val", "test"]:
            setattr(self, f"dataloader_{split}", data.dataloaders[split])
        self.num_labels = data.num_tasks

    def print_r0(self, payload):
        if self.accelerator.is_main_process:
            print(f"{payload}")

    def _initialize_optimizer(self):
        optimizer_class = getattr(torch.optim, self.cfg.optimizer.name)
        return optimizer_class(self.model.parameters(), **OmegaConf.to_container(self.cfg.optimizer.args))

    def _get_loss_function(self):
        if self.custom_loss_func:
            self.print_r0(f"> Using Custom Loss Function: {self.custom_loss_func.__name__}")
            return self.custom_loss_func
        elif self.cfg.loss.name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif self.cfg.loss.name == "BCEWithLogitsLoss":
            return torch.nn.BCEWithLogitsLoss()
        elif self.cfg.loss.name == "MaskedBCEWithLogitsLoss":
            return Heimdall.losses.MaskedBCEWithLogitsLoss()
        elif self.cfg.loss.name == "CrossEntropyFocalLoss":
            return Heimdall.losses.CrossEntropyFocalLoss()
        elif self.cfg.loss.name == "MSELoss":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.cfg.loss.name}")

    def _initialize_wandb(self):
        if self.run_wandb and self.accelerator.is_main_process:
            print("==> Starting a new WANDB run")
            new_tags = (self.cfg.dataset.dataset_name, self.cfg.fg.type, self.cfg.fe.type, self.cfg.fc.type)
            wandb_config = {
                "wandb": {
                    "tags": new_tags,
                    "name": self.cfg.run_name,
                    "entity": self.cfg.entity,
                },
            }
            self.accelerator.init_trackers(
                project_name=self.cfg.project_name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                init_kwargs=wandb_config,
            )
            print("==> Initialized Run")

    def _initialize_lr_scheduler(self):
        dataset_config = self.cfg.tasks.args
        global_batch_size = dataset_config.batchsize
        total_steps = len(self.dataloader_train.dataset) // global_batch_size * dataset_config.epochs
        warmup_ratio = self.cfg.scheduler.warmup_ratio
        warmup_step = int(warmup_ratio * total_steps)

        self.lr_scheduler = get_scheduler(
            name=self.cfg.scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=total_steps,
        )
        self.print_r0("!!! Remember that config batchsize here is GLOBAL Batchsize !!!")
        self.print_r0(f"> global batchsize: {global_batch_size}")
        self.print_r0(f"> total_samples: {len(self.dataloader_train.dataset)}")
        self.print_r0(f"> Warm Up Steps: {warmup_step}")
        self.print_r0(f"> Total Steps: {total_steps}")
        self.print_r0(f"> per_device_batch_size: {global_batch_size // self.accelerator.num_processes}")

    def _initialize_metrics(self):
        """Initializing the metrics based on the hydra config."""
        metrics = {}
        task_type = self.cfg.tasks.args.task_type

        # First, add custom metrics if provided, TODO this is not implemented yet
        assert self.custom_metrics == {}, "Custom Metrics Not Implemented Yet"
        metrics.update(self.custom_metrics)

        # Then, add built-in metrics if not overridden by custom metrics
        if task_type == "multiclass":
            num_classes = self.num_labels
            for metric_name in self.cfg.tasks.args.metrics:
                if metric_name not in metrics:
                    if metric_name == "Accuracy":
                        metrics[metric_name] = Accuracy(task="multiclass", num_classes=num_classes)
                    elif metric_name == "Precision":
                        metrics[metric_name] = Precision(task="multiclass", num_classes=num_classes, average="macro")
                    elif metric_name == "Recall":
                        metrics[metric_name] = Recall(task="multiclass", num_classes=num_classes, average="macro")
                    elif metric_name == "F1Score":
                        metrics[metric_name] = F1Score(task="multiclass", num_classes=num_classes, average="macro")
                    elif metric_name == "MatthewsCorrCoef":
                        metrics[metric_name] = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
                    elif metric_name == "ConfusionMatrix":
                        metrics[metric_name] = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        elif task_type == "regression":
            for metric_name in self.cfg.tasks.args.metrics:
                if metric_name not in metrics:
                    if metric_name == "R2Score":
                        metrics[metric_name] = R2Score()
                    elif metric_name == "MSE":
                        metrics[metric_name] = MeanSquaredError()
        elif task_type == "binary":
            # num_labels = self.num_labels
            num_labels = 2
            for metric_name in self.cfg.tasks.args.metrics:
                if metric_name not in metrics:
                    if metric_name == "Accuracy":
                        metrics[metric_name] = Accuracy(task="binary", num_labels=num_labels)
                    elif metric_name == "Precision":
                        metrics[metric_name] = Precision(task="binary", num_labels=num_labels, average="macro")
                    elif metric_name == "Recall":
                        metrics[metric_name] = Recall(task="binary", num_labels=num_labels, average="macro")
                    elif metric_name == "F1Score":
                        metrics[metric_name] = F1Score(task="binary", num_labels=num_labels, average="macro")
                    elif metric_name == "MatthewsCorrCoef":
                        metrics[metric_name] = MatthewsCorrCoef(task="binary", num_labels=num_labels)

        return {k: v.to(self.accelerator.device) if hasattr(v, "to") else v for k, v in metrics.items()}

    def fit(self, resume_from_checkpoint=True, checkpoint_every_n_epochs=1):
        """Train the model with automatic checkpointing and resumption."""
        # Initialize checkpointing
        self.initialize_checkpointing()

        # Try to resume from checkpoint if requested
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint()

        # If the tracked parameter is specified
        track_metric = None
        if self.cfg.tasks.args.get("track_metric", False):
            track_metric = self.cfg.tasks.args.track_metric
            best_metric = {
                f"best_val_{track_metric}": float("-inf"),
                f"reported_test_{track_metric}": float("-inf"),
            }
            assert (
                track_metric in self.cfg.tasks.args.metrics
            ), "The tracking metric is not in the list of metrics, please check your configuration task file"

        # Initialize early stopping parameters
        early_stopping = self.cfg.tasks.args.get("early_stopping", False)
        early_stopping_patience = self.cfg.tasks.args.get("early_stopping_patience", 5)
        patience_counter = 0

        best_val_embed = None
        best_test_embed = None
        best_epoch = 0

        for epoch in range(start_epoch, self.cfg.tasks.args.epochs):
            # Validation and test evaluation
            valid_log, val_embed = self.validate_model(self.dataloader_val, dataset_type="valid")
            test_log, test_embed = self.validate_model(self.dataloader_test, dataset_type="test")

            # Track the best metric if specified
            if track_metric:
                val_metric = valid_log.get(f"valid_{track_metric}", float("-inf"))
                if val_metric > best_metric[f"best_val_{track_metric}"]:

                    best_val_embed = val_embed
                    best_test_embed = test_embed
                    best_epoch = epoch

                    best_metric[f"best_val_{track_metric}"] = val_metric
                    self.print_r0(f"New best validation {track_metric}: {val_metric}")
                    best_metric["reported_epoch"] = epoch  # log the epoch for convenience
                    for metric in self.cfg.tasks.args.metrics:
                        best_metric[f"reported_test_{metric}"] = test_log.get(f"test_{metric}", float("-inf"))
                    patience_counter = 0  # Reset patience counter since we have a new best

                    # Save checkpoint for best model
                    self.save_checkpoint(epoch)
                    self.print_r0(f"> Saved best model checkpoint at epoch {epoch}")
                else:
                    patience_counter += 1
                    if early_stopping:
                        self.print_r0(
                            f"No improvement in validation {track_metric}. "
                            f"Patience counter: {patience_counter}/{early_stopping_patience}",
                        )

            # Check early stopping condition
            if early_stopping and patience_counter >= early_stopping_patience:
                self.print_r0(
                    f"Early stopping triggered. No improvement in {track_metric} for {early_stopping_patience} epochs.",
                )
                break

            # Train for one epoch
            self.train_epoch(epoch)

            # Save checkpoint at regular intervals if requested
            if (epoch + 1) % checkpoint_every_n_epochs == 0:
                self.save_checkpoint(epoch)
                self.print_r0(f"> Saved regular checkpoint at epoch {epoch}")

        # # Save final checkpoint ## no need to save the final checkpoint
        # self.save_checkpoint(epoch)
        # self.print_r0(f"> Saved final checkpoint at epoch {epoch}")

        if self.run_wandb and self.accelerator.is_main_process:
            if track_metric:  # logging the best val score and the tracked test scores
                self.accelerator.log(best_metric, step=self.step)
            self.accelerator.end_training()

        if (
            self.accelerator.is_main_process
            and self.cfg.model.name != "logistic_regression"
            and not isinstance(self.data.datasets["full"], Heimdall.datasets.PairedInstanceDataset)
        ):
            self.save_adata_umap(best_test_embed, best_val_embed)
            self.print_r0(f"> Saved best UMAP checkpoint at epoch {best_epoch}")

        if self.accelerator.is_main_process:
            self.print_r0("> Model has finished Training")

    def get_loss(self, logits, labels, masks=None):
        if masks is not None:
            logits, labels = logits[masks], labels[masks]

        if self.custom_loss_func:
            loss = self.loss_fn(logits, labels)
        elif self.cfg.loss.name.endswith("BCEWithLogitsLoss"):
            loss = self.loss_fn(logits, labels)
        elif self.cfg.loss.name.endswith("CrossEntropyFocalLoss"):
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.cfg.loss.name == "CrossEntropyLoss":
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.cfg.loss.name == "MSELoss":
            loss = self.loss_fn(logits, labels)
        else:
            raise NotImplementedError("Only custom, CrossEntropyLoss, and MSELoss are supported right now")

        return loss

    def train_epoch(self, epoch):
        self.model.train()
        step = len(self.dataloader_train) * epoch
        log_every = 1

        with tqdm(self.dataloader_train, disable=not self.accelerator.is_main_process) as t:
            for batch in t:
                step += 1
                is_logging = step % log_every == 0

                lr = self.lr_scheduler.get_last_lr()[0]
                with self.accelerator.accumulate(self.model):

                    inputs = (batch["identity_inputs"], batch["expression_inputs"])
                    outputs = self.model(
                        inputs=inputs,
                        conditional_tokens=batch.get("conditional_tokens"),
                        attention_mask=batch.get("expression_padding"),
                    )
                    labels = batch["labels"].to(outputs.device)
                    if (masks := batch.get("masks")) is not None:
                        masks = masks.to(outputs.device)

                    loss = self.get_loss(outputs.logits, labels, masks=masks)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.trainer.grad_norm_clip,
                        )
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self.step += 1

                t.set_description(
                    f"Epoch: {epoch} "
                    f"Step {self.step} "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {lr:.1e} "
                    f"grad_norm: {grad_norm:.4f} ",
                )

                if is_logging:
                    log = {
                        "train_loss": loss.item(),
                        "global_step": self.step,
                        "learning_rate": lr,
                        "epoch": epoch,
                        "grad_norm": grad_norm,
                    }
                    if self.run_wandb and self.accelerator.is_main_process:
                        self.accelerator.log(log, step=self.step)

                if self.cfg.trainer.fastdev:
                    break

    # Add these methods to the HeimdallTrainer class
    def save_adata_umap(self, best_test_embed, best_val_embed):
        # Case 1: predefined splits
        if hasattr(self.cfg.tasks.args, "splits"):
            test_adata = self.data.adata[
                self.data.adata.obs[self.cfg.tasks.args.splits.col] == self.cfg.tasks.args.splits.keys_.test
            ].copy()
            val_adata = self.data.adata[
                self.data.adata.obs[self.cfg.tasks.args.splits.col] == self.cfg.tasks.args.splits.keys_.val
            ].copy()

        # Case 2: random splits
        elif hasattr(self.data, "splits"):
            # breakpoint()
            test_adata = self.data.adata[self.splits["test"]].copy()
            val_adata = self.data.adata[self.splits["val"]].copy()

        else:
            raise ValueError("No split information found in config")

        test_adata.obsm["heimdall_latents"] = best_test_embed
        val_adata.obsm["heimdall_latents"] = best_val_embed

        sc.pp.neighbors(test_adata, use_rep="heimdall_latents")
        sc.tl.leiden(test_adata)
        sc.tl.umap(test_adata)

        sc.pp.neighbors(val_adata, use_rep="heimdall_latents")
        sc.tl.leiden(val_adata)
        sc.tl.umap(val_adata)

        AnnData.write(test_adata, self.results_folder / "test_adata.h5ad")
        AnnData.write(val_adata, self.results_folder / "val_adata.h5ad")

    def initialize_checkpointing(self, results_folder_path=None):
        """Initialize checkpoint directory."""
        if results_folder_path is None:
            self.results_folder = Path(self.cfg.work_dir)
        else:
            self.results_folder = Path(results_folder_path)

        # Create directory if it doesn't exist
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(parents=True, exist_ok=True)
            self.print_r0(f"> Checkpoint directory initialized at {self.results_folder}")

    def save_checkpoint(self, epoch):
        """Save model checkpoint at the given epoch."""
        # Only save on the main process
        if not self.accelerator.is_main_process:
            return

        # Ensure results folder exists
        if not hasattr(self, "results_folder"):
            self.initialize_checkpointing()

        # Calculate current step based on epoch
        # step = len(self.dataloader_train) * epoch

        # Prepare the data to save
        data = {
            "epoch": epoch,
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if (self.accelerator.scaler is not None) else None,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "version": 1.0,
        }

        # Save checkpoint
        checkpoint_path = self.results_folder / f"model-{epoch}.pt"
        torch.save(data, str(checkpoint_path))
        self.print_r0(f"> Saved checkpoint to {checkpoint_path}")

        # Overwrite 'milestone.txt' with the new milestone
        milestone_file = self.results_folder / "milestone.txt"
        with open(milestone_file, "w") as f:
            f.write(str(epoch))
        self.print_r0(f"> Updated milestone.txt to milestone {epoch}")

        config_path = self.results_folder / "config.txt"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    def load_checkpoint(self, specific_milestone=None):
        """Load a checkpoint based on milestone.txt or a specific milestone
        number."""
        # Ensure results folder is initialized
        if not hasattr(self, "results_folder"):
            self.initialize_checkpointing()

        if not self.results_folder.exists():
            self.print_r0(f"> Results folder {self.results_folder} does not exist. Starting from scratch.")
            return 0

        # Determine which milestone to load
        if specific_milestone is not None:
            milestone = specific_milestone
        else:
            milestone_file = self.results_folder / "milestone.txt"
            if not milestone_file.exists():
                self.print_r0("> No milestone.txt found. Starting from scratch.")
                return 0

            # Read the milestone number
            with open(milestone_file) as f:
                milestone_str = f.read().strip()
                if not milestone_str.isdigit():
                    self.print_r0("milestone.txt is invalid. Starting from scratch.")
                    return 0
                milestone = int(milestone_str)

        # Load the checkpoint
        load_path = self.results_folder / f"model-{milestone}.pt"
        if not load_path.exists():
            self.print_r0(f"> Checkpoint file {load_path} does not exist. Starting from scratch.")
            return 0

        self.print_r0(f"> Loading checkpoint from {load_path}")

        # Load the data
        device = self.accelerator.device
        data = torch.load(str(load_path), map_location=device, weights_only=False)

        # Unwrap model and restore parameters
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        # Restore optimizer and scheduler states
        self.optimizer.load_state_dict(data["optimizer"])
        if (data["scaler"] is not None) and (self.accelerator.scaler is not None):
            self.accelerator.scaler.load_state_dict(data["scaler"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

        # Restore random states
        random.setstate(data["python_rng_state"])
        np.random.set_state(data["numpy_rng_state"])

        # Handle torch RNG state
        torch_rng_state = data["torch_rng_state"]
        if isinstance(torch_rng_state, torch.Tensor) and torch_rng_state.device.type != "cpu":
            torch_rng_state = torch_rng_state.cpu()
        torch.random.set_rng_state(torch_rng_state)

        # Handle CUDA RNG states
        if data["cuda_rng_state_all"] is not None and torch.cuda.is_available():
            num_visible_devices = torch.cuda.device_count()
            if len(data["cuda_rng_state_all"]) != num_visible_devices:
                self.print_r0(
                    "Warning: Number of visible CUDA devices does not match the number of saved CUDA RNG states. "
                    "Skipping CUDA RNG state restoration.",
                )
            else:
                new_cuda_states = []
                for state in data["cuda_rng_state_all"]:
                    if isinstance(state, torch.Tensor) and state.device.type != "cpu":
                        state = state.cpu()
                    new_cuda_states.append(state)
                torch.cuda.set_rng_state_all(new_cuda_states)

        epoch = data["epoch"]
        self.step = data["step"]
        # step = data.get("step", epoch * len(self.dataloader_train))

        if "version" in data:
            self.print_r0(f"> Checkpoint version: {data['version']}")
        self.print_r0(f"> Resumed from epoch {epoch}, step {self.step}")

        return epoch + 1  # Return the next epoch to start from

    def validate_model(self, dataloader, dataset_type):
        self.model.eval()
        metrics = self._initialize_metrics()
        # print(metrics)
        loss = 0
        encoded_list = []

        y_true_batches, preds_batches = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=not self.accelerator.is_main_process):
                inputs = (batch["identity_inputs"], batch["expression_inputs"])

                outputs = self.model(
                    inputs=inputs,
                    conditional_tokens=batch.get("conditional_tokens"),
                    attention_mask=batch.get("expression_padding"),
                )

                logits = outputs.logits
                labels = batch["labels"].to(outputs.device)

                if self.cfg.tasks.args.task_type == "multiclass":
                    preds = logits.argmax(1)

                elif self.cfg.tasks.args.task_type == "binary":
                    # multi-label binary classification → use sigmoid + threshold
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()

                elif self.cfg.tasks.args.task_type == "regression":
                    preds = logits

                else:
                    raise ValueError(f"Unsupported task_type: {self.cfg.tasks.args.task_type}")

                y_true_batches.append(labels.cpu())
                preds_batches.append(preds.cpu())

                if self.cfg.model.name != "logistic_regression":
                    encoded_list.append(outputs.cls_embeddings.detach().cpu().numpy())

                if (masks := batch.get("masks")) is not None:
                    masks = masks.to(outputs.device)
                    logits, labels = logits[masks], labels[masks]

                # perform a .clone() so that the labels are not updated in-place
                loss += self.get_loss(logits, labels.clone()).item()

                # predictions = outputs["logits"] if isinstance(outputs, dict) else outputs
                # labels = batch['labels']

                # print(metrics)
                # print("---")
                for metric_name, metric in metrics.items():  # noqa: B007
                    # Built-in metric
                    # print(metric)
                    # print(metric_name)
                    if self.cfg.tasks.args.task_type in ["multiclass"]:
                        labels = labels.to(torch.int)
                    if self.cfg.tasks.args.task_type in ["binary"]:
                        # Step 1: Flatten the tensor
                        flattened_labels = labels.flatten()
                        flattened_logits = logits.flatten()
                        mask = ~torch.isnan(flattened_labels)

                        no_nans_flattened_labels = flattened_labels[mask]
                        no_nans_flattened_logits = flattened_logits[mask]
                        labels = no_nans_flattened_labels.to(torch.int)
                        logits = no_nans_flattened_logits
                    metric.update(logits, labels)
                if self.cfg.trainer.fastdev:
                    break

        all_encoded = None
        if self.cfg.model.name != "logistic_regression":
            all_encoded = np.concatenate(encoded_list, axis=0)

        loss = loss / len(dataloader)

        # concatenate & gather once per epoch
        y_true_all = torch.cat(y_true_batches, 0)
        preds_all = torch.cat(preds_batches, 0)

        if self.accelerator.num_processes > 1:
            loss = self.accelerator.gather(torch.tensor(loss)).mean().item()

        log = {f"{dataset_type}_loss": loss}
        for metric_name, metric in metrics.items():
            if metric_name != "ConfusionMatrix":
                # Built-in metric
                log[f"{dataset_type}_{metric_name}"] = metric.compute().item()
                if metric_name in ["Accuracy", "Precision", "Recall", "F1Score", "MathewsCorrCoef"]:
                    log[f"{dataset_type}_{metric_name}"] *= 100  # Convert to percentage for these metrics

        if "ConfusionMatrix" in metrics:
            # 1. Gather counts from all processes and sum
            cm_local = metrics["ConfusionMatrix"].compute()  # (C, C) tensor
            cm_counts = self.accelerator.reduce(cm_local, reduction="sum")  # global counts

            # 3) If binary and flat, reshape to (2, 2)
            if cm_counts.dim() == 1:
                c = int(cm_counts.numel() ** 0.5)  # should be 2
                cm_counts = cm_counts.view(c, c)

            # 2. Row-wise normalisation → per-class accuracy matrix
            cm_norm = cm_counts.float()
            cm_norm = cm_norm / (cm_norm.sum(dim=1, keepdim=True) + 1e-8)

            # 3. Per-class accuracy vector (for dashboard scalars)
            per_class_acc = cm_norm.diag().cpu().numpy() * 100
            log[f"{dataset_type}_per_class_accuracy"] = {
                name: float(acc) for name, acc in zip(self.class_names, per_class_acc)
            }

            # 4. Log interactive confusion matrix to WandB (main process only)
            if self.run_wandb and self.accelerator.is_main_process:
                wandb_cm = wandb.plot.confusion_matrix(
                    y_true=y_true_all.numpy().tolist(),
                    preds=preds_all.numpy().tolist(),
                    class_names=self.class_names,  # same order as metric
                )
                self.accelerator.log(
                    {f"{dataset_type}_confusion_matrix": wandb_cm},
                    step=self.step,
                )

        rss = self.process.memory_info().rss / (1024**3)
        log["Process_mem_rss"] = rss

        if self.run_wandb and self.accelerator.is_main_process:
            self.accelerator.log(log, step=self.step)

        if not self.run_wandb and self.accelerator.is_main_process:
            print(log)

        return log, all_encoded
