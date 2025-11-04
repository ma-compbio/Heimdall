"""Heimdall trainer."""

import random
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat

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
from Heimdall.utils import (
    INPUT_KEYS,
    get_dtype,
    instantiate_from_config,
    project2simplex_,
    save_umap,
)


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
        self.has_embeddings = self.cfg.model.name != "logistic_regression"

        self.check_flash_attn()

        # TODO: since we use the label_key in the CellRepresentation setup, we shouldn't need it here.
        # It should all be accessible in the data.labels... Delete the block below if possible...?

        # Unified label key handling: support .obs or .obsm
        self.class_names = {}
        for subtask_name, subtask in self.data.tasklist:
            label_key = subtask.label_col_name
            label_obsm_key = subtask.label_obsm_name

            if subtask.task_type in ("multiclass", "binary"):
                if label_key is not None:
                    # Single-label classification using .obs[label_key]
                    if not pd.api.types.is_categorical_dtype(self.data.adata.obs[label_key]):
                        self.data.adata.obs[label_key] = self.data.adata.obs[label_key].astype("category")
                    self.class_names[subtask_name] = self.data.adata.obs[label_key].cat.categories.tolist()
                elif label_obsm_key is not None:
                    self.class_names[subtask_name] = self.data.adata.obsm[label_obsm_key].columns.tolist()
                else:
                    self.class_names[subtask_name] = data.adata.uns["task_order"]  # NOTE: first entry might be NULL

        self.num_labels = {}
        for subtask_name, subtask in self.data.tasklist:
            label_key = subtask.label_col_name
            label_obsm_key = subtask.label_obsm_name
            if subtask.task_type in ("multiclass", "binary") and (label_key or label_obsm_key):
                self.num_labels[subtask_name] = len(self.class_names[subtask_name])
            else:
                self.num_labels[subtask_name] = subtask.num_tasks

        self.run_wandb = run_wandb
        self.process = psutil.Process()
        self.custom_loss_func = custom_loss_func
        self.custom_metrics = custom_metrics or {}

        set_seed(cfg.seed)

        self.optimizer = self._initialize_optimizer()
        self.loss_functions = self.instantiate_loss_functions_from_config()

        self.accelerator.wait_for_everyone()
        self.print_r0(f"> Using Device: {self.accelerator.device}")
        self.print_r0(f"> Number of Devices: {self.accelerator.num_processes}")

        self.best_val_embed = {}
        self.best_test_embed = {}
        self.best_epoch = defaultdict(int)

        self._initialize_wandb()
        self._initialize_lr_scheduler()
        self.step = 0

        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )

        self.print_r0("> Finished Wrapping the model, optimizer, and dataloaders in accelerate")
        self.print_r0("> run HeimdallTrainer.train() to begin training")

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

    def print_r0(self, message):
        self.data.print_r0(message)

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
        tasklist = self.data.tasklist
        global_batch_size = tasklist.batchsize
        total_steps = len(self.dataloader_train.dataset) // global_batch_size * tasklist.epochs
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
        metrics = defaultdict(dict)
        for subtask_name, subtask in self.data.tasklist:
            subtask_metrics = metrics[subtask_name]
            task_type = subtask.task_type

            # First, add custom metrics if provided, TODO this is not implemented yet
            assert self.custom_metrics == {}, "Custom metrics not implemented yet"
            subtask_metrics.update(self.custom_metrics)

            # Then, add built-in metrics if not overridden by custom metrics
            if task_type in ("mlm", "multiclass"):
                num_classes = self.num_labels[subtask_name]
                for metric_name in subtask.metrics:
                    if metric_name not in metrics:
                        if metric_name == "Accuracy":
                            subtask_metrics[metric_name] = Accuracy(task="multiclass", num_classes=num_classes)
                            if subtask.top_k is not None:
                                for k in subtask.top_k:
                                    subtask_metrics[f"{metric_name}_top_{k}"] = Accuracy(
                                        task="multiclass",
                                        num_classes=num_classes,
                                        top_k=k,
                                    )

                        elif metric_name == "Precision":
                            subtask_metrics[metric_name] = Precision(
                                task="multiclass",
                                num_classes=num_classes,
                                average="macro",
                            )
                        elif metric_name == "Recall":
                            subtask_metrics[metric_name] = Recall(
                                task="multiclass",
                                num_classes=num_classes,
                                average="macro",
                            )
                        elif metric_name == "F1Score":
                            subtask_metrics[metric_name] = F1Score(
                                task="multiclass",
                                num_classes=num_classes,
                                average="macro",
                            )
                        elif metric_name == "MatthewsCorrCoef":
                            subtask_metrics[metric_name] = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
                        elif metric_name == "ConfusionMatrix" and task_type != "mlm":
                            subtask_metrics[metric_name] = ConfusionMatrix(task="multiclass", num_classes=num_classes)
            elif task_type == "regression":
                for metric_name in subtask.metrics:
                    if metric_name not in metrics:
                        if metric_name == "R2Score":
                            subtask_metrics[metric_name] = R2Score()
                        elif metric_name == "MSE":
                            subtask_metrics[metric_name] = MeanSquaredError()
            elif task_type == "binary":
                # num_labels = self.num_labels
                num_labels = 2
                for metric_name in subtask.metrics:
                    if metric_name not in metrics:
                        if metric_name == "Accuracy":
                            subtask_metrics[metric_name] = Accuracy(task="binary", num_labels=num_labels)
                        elif metric_name == "Precision":
                            subtask_metrics[metric_name] = Precision(
                                task="binary",
                                num_labels=num_labels,
                                average="macro",
                            )
                        elif metric_name == "Recall":
                            subtask_metrics[metric_name] = Recall(task="binary", num_labels=num_labels, average="macro")
                        elif metric_name == "F1Score":
                            subtask_metrics[metric_name] = F1Score(
                                task="binary",
                                num_labels=num_labels,
                                average="macro",
                            )
                        elif metric_name == "MatthewsCorrCoef":
                            subtask_metrics[metric_name] = MatthewsCorrCoef(task="binary", num_labels=num_labels)

            metrics[subtask_name] = {
                k: v.to(self.accelerator.device) if hasattr(v, "to") else v for k, v in subtask_metrics.items()
            }

        return metrics

    def fit(self, resume_from_checkpoint=True, checkpoint_every_n_epochs=1):
        """Train the model with automatic checkpointing and resumption."""
        # Initialize checkpointing
        self.initialize_checkpointing()

        # Try to resume from checkpoint if requested
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint()

        if start_epoch >= self.data.tasklist.epochs:
            # last_epoch = max(0, start_epoch - 1)
            # Run one eval pass on the loaded weights to get embeddings
            _, val_embed = self.validate_model(self.dataloader_val, "valid")
            _, test_embed = self.validate_model(self.dataloader_test, "test")
            if self.accelerator.is_main_process and self.cfg.model.name != "logistic_regression":
                # self.save_adata_umap(test_embed, val_embed)
                # self.print_r0(f"> Saved UMAP from checkpoint epoch {last_epoch}")
                pass
            return

        # If the tracked parameter is specified
        track_metric = defaultdict(lambda: None)
        best_metric = defaultdict(dict)
        for subtask_name, subtask in self.data.tasklist:
            if subtask.track_metric is not None:
                track_metric[subtask_name] = subtask.track_metric
                best_metric[subtask_name] = {
                    f"best_val_{subtask_name}_{track_metric}": float("-inf"),
                    f"reported_test_{subtask_name}_{track_metric}": float("-inf"),
                }
                assert (
                    track_metric[subtask_name] in subtask.metrics
                ), "The tracking metric is not in the list of metrics, please check your configuration task file"

        # Initialize early stopping parameters
        early_stopping = self.data.tasklist.early_stopping
        early_stopping_patience = self.data.tasklist.early_stopping_patience
        patience_counter = 0

        for epoch in range(start_epoch, self.data.tasklist.epochs):
            # Validation and test evaluation
            valid_log, val_embed = self.validate_model(self.dataloader_val, dataset_type="valid")
            test_log, test_embed = self.validate_model(self.dataloader_test, dataset_type="test")

            # Track the best metric if specified
            reset_patience_counter = False
            for subtask_name, subtask in self.data.tasklist:
                if track_metric[subtask_name] is not None:
                    val_metric = valid_log.get(f"valid_{subtask_name}_{track_metric}", float("-inf"))
                    if (
                        val_metric > best_metric[subtask_name][f"best_val_{subtask_name}_{track_metric}"]
                    ):  # Change to >= if you want to debug UMAP
                        self.best_val_embed[subtask_name] = val_embed[subtask_name]
                        self.best_test_embed[subtask_name] = test_embed[subtask_name]
                        self.best_epoch[subtask_name] = epoch

                        best_metric[subtask_name][f"best_val_{subtask_name}_{track_metric}"] = val_metric
                        self.print_r0(f"New best validation for {subtask_name} {track_metric}: {val_metric}")
                        best_metric[subtask_name]["reported_epoch"] = epoch  # log the epoch for convenience
                        for metric in subtask.metrics:
                            best_metric[subtask_name][f"reported_test_{metric}"] = test_log.get(
                                f"test_{metric}",
                                float("-inf"),
                            )

                        reset_patience_counter = True

                        # Save checkpoint for best model
                        self.save_checkpoint(epoch)
                        self.print_r0(f"> Saved best model checkpoint at epoch {epoch}")

                else:
                    self.best_val_embed[subtask_name] = val_embed[subtask_name]
                    self.best_test_embed[subtask_name] = test_embed[subtask_name]
                    self.best_epoch[subtask_name] = epoch

                if reset_patience_counter:
                    patience_counter = 0  # Reset patience counter since we have a new best
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

        if self.run_wandb and self.accelerator.is_main_process:
            for subtask_name, _ in self.data.tasklist:
                if track_metric[subtask_name] is not None:  # logging the best val score and the tracked test scores
                    self.accelerator.log(best_metric[subtask_name], step=self.step)
            self.accelerator.end_training()

        if (
            self.accelerator.is_main_process
            and self.has_embeddings
            and not isinstance(self.data.datasets["full"], Heimdall.datasets.PairedInstanceDataset)
            # TODO doesn't seem necessary for pretraining but consult with others
        ):
            if self.best_test_embed and self.best_val_embed and not self.cfg.trainer.fastdev:
                for subtask_name, _ in self.data.tasklist:
                    save_umap(
                        self.data,
                        self.best_test_embed[subtask_name],
                        embedding_name=f"{subtask_name}_latents",
                        split="test",
                        savepath=self.results_folder / "test_adata.h5ad",
                    )
                    save_umap(
                        self.data,
                        self.best_val_embed[subtask_name],
                        embedding_name=f"{subtask_name}_latents",
                        split="val",
                        savepath=self.results_folder / "val_adata.h5ad",
                    )
                    self.print_r0(f"> Saved best UMAP checkpoint at epoch {self.best_epoch}")
            else:
                self.print_r0("> Skipped saving UMAP")

        if self.accelerator.is_main_process:
            self.print_r0("> Model has finished Training")

    def instantiate_loss_functions_from_config(self):
        loss_functions = {}
        for subtask_name, subtask in self.data.tasklist:
            loss_kwargs = {}
            loss_name = subtask.loss_config.type.split(".")[-1]
            if loss_name.startswith("Flatten"):
                loss_kwargs["num_labels"] = self.num_labels[subtask_name]

            loss_functions[subtask_name] = instantiate_from_config(subtask.loss_config, **loss_kwargs)

        return loss_functions

    def get_outputs_and_loss(self, batch, loss=None):
        for values in batch.values():
            for subtask_name, value in values.items():
                if value is not None:
                    if isinstance(value, list):
                        value = [subvalue.to(self.accelerator.device) for subvalue in value]
                    else:
                        value = value.to(self.accelerator.device)

                    values[subtask_name] = value

        inputs = {input_key: batch[input_key] for input_key in INPUT_KEYS if input_key in batch}

        # inputs = (batch["identity_inputs"], batch["expression_inputs"])

        outputs = self.model(inputs=inputs)

        batch_loss = 0
        preds = {}
        labels = batch["labels"]
        for subtask_name, subtask in self.data.tasklist:
            logits = outputs[subtask_name].logits
            subtask_labels = labels[subtask_name]

            if subtask.task_type in ("multiclass", "regression"):
                subtask_preds = logits
                # logits.argmax(dim=1)
            elif subtask.task_type == "mlm":
                subtask_preds = logits.argmax(dim=2)
            elif subtask.task_type == "binary":
                # multi-label binary classification → use sigmoid + threshold
                probs = torch.sigmoid(logits)
                subtask_preds = (probs > 0.5).float()
            else:
                raise ValueError(f"Unsupported task_type: {subtask.task_type}")

            preds[subtask_name] = subtask_preds

            if (masks := batch["masks"][subtask_name]) is not None:
                logits, subtask_labels = logits[masks], subtask_labels[masks]

            # perform a .clone() so that the subtask_labels are not updated in-place
            # TODO: weight task-specific loss_functions somehow
            loss_function = self.loss_functions[subtask_name]
            batch_loss += loss_function(logits, subtask_labels.clone())

        if loss is None:
            loss = batch_loss
        else:
            loss += batch_loss

        return outputs, labels, preds, loss

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

        constrained_params = [p for name, p in self.model.named_parameters() if "metafeature" in name]
        if training:
            step = len(dataloader) * epoch
            outputs = None
        else:
            step = 0

            outputs = {
                "all_embeddings": defaultdict(list),
                "all_labels": defaultdict(list),
                "all_preds": defaultdict(list),
            }

        with tqdm(dataloader, disable=not self.accelerator.is_main_process) as pbar:
            for batch in pbar:
                step += 1

                is_logging = step % log_every == 0
                lr = self.lr_scheduler.get_last_lr()[0]

                with self.accelerator.accumulate(self.model) if training else nullcontext():
                    batch_outputs, labels, preds, loss = self.get_outputs_and_loss(batch, loss)

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

                            with torch.no_grad():
                                for param in constrained_params:
                                    # TODO: make this more robust to different types of PGD
                                    project2simplex_(param, dim=0)

                        loss = None
                    else:
                        for subtask_name, _ in self.data.tasklist:
                            if self.has_embeddings:
                                outputs["all_embeddings"][subtask_name].append(
                                    batch_outputs[subtask_name].cls_embeddings.detach().cpu().numpy(),
                                )

                            outputs["all_labels"][subtask_name].append(labels[subtask_name].detach().cpu().numpy())
                            outputs["all_preds"][subtask_name].append(preds[subtask_name].detach().cpu().numpy())

                        outputs["loss"] = loss

                    if metrics is not None:
                        for subtask_name, subtask in self.data.tasklist:
                            for metric_name, metric in metrics[subtask_name].items():  # noqa: B007
                                # Built-in metric
                                subtask_labels = labels[subtask_name]
                                subtask_preds = preds[subtask_name]

                                if subtask.task_type in ["multiclass", "mlm"]:
                                    subtask_labels = subtask_labels.to(torch.int)

                                # Remove negative MLM values (TODO: fix so UCE doesn't provide these)
                                if subtask.task_type in ["mlm"]:
                                    if torch.any(subtask_labels < 0):
                                        flattened_labels = subtask_labels.flatten()
                                        flattened_preds = subtask_preds.flatten()
                                        mask = flattened_labels >= 0
                                        nonnegative_flattened_labels = flattened_labels[mask]
                                        nonnegative_flattened_preds = flattened_preds[mask]
                                        subtask_labels = nonnegative_flattened_labels.to(torch.int)
                                        subtask_preds = nonnegative_flattened_preds

                                # Remove NaN values
                                if subtask.task_type in ["binary"]:
                                    # Step 1: Flatten the tensor
                                    flattened_labels = subtask_labels.flatten()
                                    flattened_preds = subtask_preds.flatten()
                                    mask = ~torch.isnan(flattened_labels)

                                    no_nans_flattened_labels = flattened_labels[mask]
                                    no_nans_flattened_preds = flattened_preds[mask]
                                    subtask_labels = no_nans_flattened_labels.to(torch.int)
                                    subtask_preds = no_nans_flattened_preds

                                metric.update(subtask_preds, subtask_labels)

                if self.cfg.trainer.fastdev:
                    break

        if not training:
            for subtask_name, _ in self.data.tasklist:
                outputs["all_embeddings"][subtask_name] = np.concatenate(
                    outputs["all_embeddings"][subtask_name],
                    axis=0,
                )
                outputs["all_labels"][subtask_name] = np.concatenate(outputs["all_labels"][subtask_name], axis=0)
                outputs["all_preds"][subtask_name] = np.concatenate(outputs["all_preds"][subtask_name], axis=0)

        return outputs

    def validate_model(self, dataloader, dataset_type):
        self.model.eval()
        metrics = self._initialize_metrics()
        loss = torch.tensor(0, dtype=get_dtype(self.data._cfg.float_dtype), device=self.accelerator.device)

        if len(dataloader) == 0:
            raise ValueError("`DataLoader` length cannot be zero. Check custom sampler implementation.")

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

        for subtask_name, subtask in self.data.tasklist:
            for metric_name, metric in metrics[subtask_name].items():
                if metric_name != "ConfusionMatrix":
                    # Built-in metric
                    log[f"{dataset_type}_{subtask_name}_{metric_name}"] = metric.compute().item()
                    if metric_name.startswith(("Accuracy", "Precision", "Recall", "F1Score", "MathewsCorrCoef")):
                        log[
                            f"{dataset_type}_{subtask_name}_{metric_name}"
                        ] *= 100  # Convert to percentage for these metrics

            if subtask.top_k is not None:
                if self.run_wandb and self.accelerator.is_main_process:
                    top_k_accuracies = []
                    for k in subtask.top_k:
                        top_k_accuracies.append(log[f"{dataset_type}_{subtask_name}_Accuracy_top_{k}"])

                    tbl = wandb.Table(data=list(zip(subtask.top_k, top_k_accuracies)), columns=["k", "topk_acc"])
                    chart = wandb.plot.line(
                        tbl,
                        "k",
                        "topk_acc",
                        title=f"{dataset_type}_{subtask_name}: Top-k Accuracy",
                    )
                    self.accelerator.log({f"{dataset_type}_{subtask_name}_topk_acc_curve": chart}, step=self.step)

            if "ConfusionMatrix" in metrics[subtask_name]:
                # 1. Gather counts from all processes and sum
                cm_local = metrics[subtask_name]["ConfusionMatrix"].compute()  # (C, C) tensor
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
                log[f"{dataset_type}_{subtask_name}_per_class_accuracy"] = {
                    name: float(acc) for name, acc in zip(self.class_names, per_class_acc)
                }

                # 4. Log interactive confusion matrix to WandB (main process only)
                if self.run_wandb and self.accelerator.is_main_process:
                    wandb_cm = wandb.plot.confusion_matrix(
                        y_true=outputs["all_labels"][subtask_name],
                        preds=outputs["all_preds"][subtask_name],
                        class_names=self.class_names[subtask_name],  # same order as metric
                    )
                    self.accelerator.log(
                        {f"{dataset_type}_{subtask_name}_confusion_matrix": wandb_cm},
                        step=self.step,
                    )

        rss = self.process.memory_info().rss / (1024**3)
        log["Process_mem_rss"] = rss

        if self.run_wandb and self.accelerator.is_main_process:
            self.accelerator.log(log, step=self.step)

        if not self.run_wandb and self.accelerator.is_main_process:
            print(f"{dataset_type}_log = {pformat(log)}")

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
    if "pretrained_ckpt_path" in config:
        if not Path(config.pretrained_ckpt_path).is_file():
            raise FileNotFoundError(f"{config.pretrained_ckpt_path=} does not exist.")

        pretrained_state_dict = torch.load(config.pretrained_ckpt_path)["model"]
        filtered_pretrained_params = OrderedDict(
            filter(lambda param_tuple: "decoder" not in param_tuple[0], pretrained_state_dict.items()),
        )  # we drop the pretrained head and load all other params

        model.load_state_dict(filtered_pretrained_params, strict=False)

        if accelerator.is_main_process:
            print(f">Finished loading pretrained params loaded from {config.pretrained_ckpt_path}")

    trainer = HeimdallTrainer(cfg=config, model=model, data=cr, accelerator=accelerator, run_wandb=run_wandb)

    return trainer
