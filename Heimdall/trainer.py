"""Heimdall trainer."""

import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score, MatthewsCorrCoef, Precision, Recall
from torchmetrics.regression import MeanSquaredError, R2Score
from tqdm import tqdm
from transformers import get_scheduler

import Heimdall.datasets
import Heimdall.losses
import wandb
from Heimdall.models import setup_experiment
from Heimdall.utils import instantiate_from_config, save_umap


class HeimdallTrainer:
    def __init__(
        self,
        cfg,
        model,
        data,
        accelerator: Accelerator,
        run_wandb=False,
        custom_loss_func=None,
        custom_metrics=None,
    ):
        self.cfg = cfg
        self.model = model
        self.data = data
        self.accelerator = accelerator

        self.check_flash_attn()

        # TODO: since we use the label_key in the CellRepresentation setup, we shouldn't need it here.
        # It should all be accessible in the data.labels... Delete the block below if possible...?

        # Unified label key handling: support .obs or .obsm
        label_key = self.data.task.label_col_name
        label_obsm_key = self.data.task.label_obsm_name

        if label_key is not None:
            # Single-label classification using .obs[label_key]
            if not pd.api.types.is_categorical_dtype(self.data.adata.obs[label_key]):
                self.data.adata.obs[label_key] = self.data.adata.obs[label_key].astype("category")
            self.class_names = self.data.adata.obs[label_key].cat.categories.tolist()
            self.num_labels = len(self.class_names)
        elif label_obsm_key is not None:
            # Multi-label classification using .obsm[label_obsm_key]
            if label_obsm_key != "mlm":
                self.class_names = self.data.adata.obsm[label_obsm_key].columns.tolist()
                self.num_labels = len(self.class_names)
            else:
                self.num_labels = data.task.num_tasks
        else:
            # Auto infering
            self.class_names = data.adata.uns["task_order"]  # NOTE: first entry might be NULL
            self.num_labels = data.task.num_tasks

        self.run_wandb = run_wandb
        self.process = psutil.Process()
        self.custom_loss_func = custom_loss_func
        self.custom_metrics = custom_metrics or {}

        set_seed(cfg.seed)

        self.optimizer = self._initialize_optimizer()
        self.loss_fn = self.instantiate_loss_from_config()

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

    def check_flash_attn(self):
        if (
            hasattr(self.model.encoder.cell_sentence_model, "use_flash_attn")
            and self.model.encoder.cell_sentence_model.use_flash_attn
            and self.accelerator.mixed_precision != "bf16"
        ):
            raise ValueError("If using Flash Attention, mixed precision must be bf16")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        for split in ["train", "val", "test", "full"]:
            setattr(self, f"dataloader_{split}", data.dataloaders[split])

    def print_r0(self, payload):
        if self.accelerator.is_main_process:
            print(f"{payload}")

    def _initialize_optimizer(self):
        optimizer_class = getattr(torch.optim, self.cfg.optimizer.name)
        return optimizer_class(self.model.parameters(), **OmegaConf.to_container(self.cfg.optimizer.args))

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
        dataset_config = self.data.task
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
        task_type = self.data.task.task_type

        # First, add custom metrics if provided, TODO this is not implemented yet
        assert self.custom_metrics == {}, "Custom metrics not implemented yet"
        metrics.update(self.custom_metrics)

        # Then, add built-in metrics if not overridden by custom metrics
        if task_type in ("mlm", "multiclass"):
            num_classes = self.num_labels
            for metric_name in self.data.task.metrics:
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
            for metric_name in self.data.task.metrics:
                if metric_name not in metrics:
                    if metric_name == "R2Score":
                        metrics[metric_name] = R2Score()
                    elif metric_name == "MSE":
                        metrics[metric_name] = MeanSquaredError()
        elif task_type == "binary":
            # num_labels = self.num_labels
            num_labels = 2
            for metric_name in self.data.task.metrics:
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
        if self.data.task.track_metric is not None:
            track_metric = self.data.task.track_metric
            best_metric = {
                f"best_val_{track_metric}": float("-inf"),
                f"reported_test_{track_metric}": float("-inf"),
            }
            assert (
                track_metric in self.data.task.metrics
            ), "The tracking metric is not in the list of metrics, please check your configuration task file"

        # Initialize early stopping parameters
        early_stopping = self.data.task.early_stopping
        early_stopping_patience = self.data.task.early_stopping_patience
        patience_counter = 0

        best_val_embed = None
        best_test_embed = None
        best_epoch = 0

        for epoch in range(start_epoch, self.data.task.epochs):
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
                    for metric in self.data.task.metrics:
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
            else:
                best_val_embed = val_embed
                best_test_embed = test_embed
                best_epoch = epoch

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

        if self.run_wandb and self.accelerator.is_main_process:
            if track_metric:  # logging the best val score and the tracked test scores
                self.accelerator.log(best_metric, step=self.step)
            self.accelerator.end_training()

        if (
            self.accelerator.is_main_process
            and self.cfg.model.name != "logistic_regression"
            and not isinstance(self.data.datasets["full"], Heimdall.datasets.PairedInstanceDataset)
            and self.data.task.task_type != "mlm"
            # TODO doesn't seem necessary for pretraining but consult with others
        ):
            if best_test_embed is not None and best_val_embed is not None and not self.cfg.trainer.fastdev:
                save_umap(self.data, best_test_embed, split="test", savepath=self.results_folder / "test_adata.h5ad")
                save_umap(self.data, best_val_embed, split="val", savepath=self.results_folder / "val_adata.h5ad")
                self.print_r0(f"> Saved best UMAP checkpoint at epoch {best_epoch}")
            else:
                self.print_r0("> Skipped saving UMAP")

        if self.accelerator.is_main_process:
            self.print_r0("> Model has finished Training")

    def instantiate_loss_from_config(self):
        loss_kwargs = {}
        loss_name = self.data.task.loss_config.type.split(".")[-1]
        if loss_name.startswith("Flatten"):
            loss_kwargs["num_labels"] = self.num_labels

        return instantiate_from_config(self.data.task.loss_config, **loss_kwargs)

    def get_loss(self, logits, labels, *args):
        if args:
            return self.loss_fn(logits, labels, *args)

        return self.loss_fn(logits, labels)

    def get_outputs_and_loss(self, batch, loss=None):
        inputs = (batch["identity_inputs"], batch["expression_inputs"])

        outputs = self.model(
            inputs=inputs,
            attention_mask=batch.get("expression_padding"),
        )

        logits = outputs.logits
        labels = batch["labels"].to(outputs.device)

        if (masks := batch.get("masks")) is not None:
            masks = masks.to(outputs.device)
            logits, labels = logits[masks], labels[masks]

        # perform a .clone() so that the labels are not updated in-place
        batch_loss = self.get_loss(logits, labels.clone())
        if loss is None:
            loss = batch_loss
        else:
            loss += batch_loss

        return outputs, loss

    def iterate_dataloader(
        self,
        dataloader,
        loss=None,
        epoch=None,
        metrics=None,
        log_every: int = 1,
    ):
        """Iterate through `DataLoader` (either for training or for
        validation)."""
        training = epoch is not None

        if training:
            step = len(dataloader) * epoch
            outputs = None
        else:
            step = 0

            outputs = {
                "all_embeddings": [],
                "all_labels": [],
                "all_preds": [],
            }

        with tqdm(dataloader, disable=not self.accelerator.is_main_process) as pbar:
            for batch in pbar:
                step += 1

                is_logging = step % log_every == 0
                lr = self.lr_scheduler.get_last_lr()[0]

                with self.accelerator.accumulate(self.model) if training else nullcontext():
                    batch_outputs, loss = self.get_outputs_and_loss(batch, loss)
                    logits = batch_outputs.logits
                    labels = batch["labels"].to(batch_outputs.device)

                    if self.data.task.task_type == "multiclass":
                        preds = logits.argmax(dim=1)
                    elif self.data.task.task_type == "mlm":
                        preds = logits.argmax(dim=2)
                    elif self.data.task.task_type == "binary":
                        # multi-label binary classification → use sigmoid + threshold
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                    elif self.data.task.task_type == "regression":
                        preds = logits
                    else:
                        raise ValueError(f"Unsupported task_type: {self.data.task.task_type}")

                    if training:
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

                            pbar.set_description(
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
                        loss = None
                    else:
                        if self.cfg.model.name != "logistic_regression":
                            outputs["all_embeddings"].append(batch_outputs.cls_embeddings.detach().cpu().numpy())

                        outputs["all_labels"].append(labels.detach().cpu().numpy())
                        outputs["all_preds"].append(preds.detach().cpu().numpy())
                        outputs["loss"] = loss

                    if metrics is not None:
                        for metric_name, metric in metrics.items():  # noqa: B007
                            # Built-in metric
                            if self.data.task.task_type in ["multiclass", "mlm"]:
                                labels = labels.to(torch.int)
                            if self.data.task.task_type in ["binary"]:
                                # Step 1: Flatten the tensor
                                flattened_labels = labels.flatten()
                                flattened_preds = preds.flatten()
                                mask = ~torch.isnan(flattened_labels)

                                no_nans_flattened_labels = flattened_labels[mask]
                                no_nans_flattened_preds = flattened_preds[mask]
                                labels = no_nans_flattened_labels.to(torch.int)
                                preds = no_nans_flattened_preds

                            metric.update(preds, labels)

                if self.cfg.trainer.fastdev:
                    break

        if not training:
            outputs["all_embeddings"] = np.concatenate(outputs["all_embeddings"], axis=0)
            outputs["all_labels"] = np.concatenate(outputs["all_embeddings"], axis=0)
            outputs["all_preds"] = np.concatenate(outputs["all_embeddings"], axis=0)

        return outputs

    def validate_model(self, dataloader, dataset_type):
        self.model.eval()
        metrics = self._initialize_metrics()
        loss = 0

        with torch.no_grad():
            outputs = self.iterate_dataloader(
                dataloader,
                loss,
                metrics=metrics,
            )
            loss += outputs["loss"]

        loss = loss / len(dataloader)

        if self.accelerator.num_processes > 1:
            loss_tensor = torch.tensor(
                [loss],
                device=self.accelerator.device,
            )
            # loss is a python floating point value, for gather
            # operation across multiple processes needs to be
            # cuda tensor
            loss = self.accelerator.gather(loss_tensor).mean().item()

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
                    y_true=outputs["all_labels"],
                    preds=outputs["all_preds"],
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

        return log, outputs["all_embeddings"]

    def train_epoch(self, epoch):
        self.model.train()

        self.iterate_dataloader(
            self.dataloader_train,
            epoch=epoch,
        )

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


def setup_trainer(config, cpu=True):
    experiment_primitives = setup_experiment(config, cpu=cpu)
    if experiment_primitives is None:
        return

    accelerator, cr, model, run_wandb = experiment_primitives
    trainer = HeimdallTrainer(cfg=config, model=model, data=cr, accelerator=accelerator, run_wandb=run_wandb)

    return trainer
