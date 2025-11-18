"""Heimdall trainer."""

import random
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Callable

import numpy as np
import pandas as pd
import psutil
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torchmetrics.classification import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
)
from torchmetrics.regression import MeanSquaredError, R2Score
from tqdm import tqdm
from transformers import get_scheduler

import Heimdall.datasets
import Heimdall.losses
import wandb
from Heimdall.cell_representations import setup_data
from Heimdall.models import TransformerOutput, setup_model
from Heimdall.utils import (  # get_cached_paths,
    INPUT_KEYS,
    get_fully_qualified_cache_paths,
    instantiate_from_config,
    project2simplex_,
    save_umap,
)

# from Heimdall.cell_representations import PartitionedCellRepresentation


class HeimdallTrainer:
    CHECKPOINT_KEYS = ("fg", "fe", "fc", "model")

    def __init__(
        self,
        cfg,
        model,
        data,
        accelerator: Accelerator,
        random_seed: int = 0,
        accumulate_grad_batches: int = 1,
        grad_norm_clip: float = 1.0,
        skip_umaps: bool = True,
        fastdev: bool = False,  # if set to true, then only train/evel/test on the first batch
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

        self.setup_class_names_and_num_labels(data)

        self.random_seed = random_seed
        self.accumulate_grad_batches = accumulate_grad_batches
        self.grad_norm_clip = grad_norm_clip
        self.skip_umaps = skip_umaps
        self.fastdev = fastdev
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

        self.best_val_outputs = defaultdict(dict)
        self.best_test_outputs = defaultdict(dict)
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

    def setup_class_names_and_num_labels(self, data):
        self.class_names = {}
        for subtask_name, subtask in data.tasklist:
            label_key = subtask.label_col_name
            label_obsm_key = subtask.label_obsm_name

            if subtask.task_type in ("multiclass", "binary"):
                if label_key is not None:
                    # Single-label classification using .obs[label_key]
                    if not pd.api.types.is_categorical_dtype(data.adata.obs[label_key]):
                        data.adata.obs[label_key] = data.adata.obs[label_key].astype("category")
                    self.class_names[subtask_name] = data.adata.obs[label_key].cat.categories.tolist()
                elif label_obsm_key is not None:
                    self.class_names[subtask_name] = data.adata.obsm[label_obsm_key].columns.tolist()
                else:
                    self.class_names[subtask_name] = data.adata.uns["task_order"]  # NOTE: first entry might be NULL

        self.num_labels = {}
        for subtask_name, subtask in data.tasklist:
            label_key = subtask.label_col_name
            label_obsm_key = subtask.label_obsm_name
            if subtask.task_type in ("multiclass", "binary") and (label_key or label_obsm_key):
                self.num_labels[subtask_name] = len(self.class_names[subtask_name])
            else:
                self.num_labels[subtask_name] = subtask.num_tasks

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        for split in ["train", "val", "test", "full"]:
            setattr(self, f"dataloader_{split}", data.dataloaders[split])

    @property
    def save_precomputed(self):
        return self.data._save_precomputed

    @save_precomputed.setter
    def save_precomputed(self, val):
        self.data._save_precomputed = val

    @property
    def get_precomputed(self):
        return self.data._get_precomputed

    @get_precomputed.setter
    def get_precomputed(self, val):
        self.data._get_precomputed = val

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

    def fit(self, get_precomputed=False, **fit_kwargs):
        context = nullcontext()
        if get_precomputed:
            context = PrecomputationContext(
                self,
                save_precomputed=False,
                get_precomputed=get_precomputed,
                run_wandb=True,
            )

        with context:
            return self.fit_model(**fit_kwargs)

    def fit_model(
        self,
        resume_from_checkpoint=True,
        checkpoint_every_n_epochs=1,
        precompute_last_epoch=False,
        do_cleanup=True,
    ):
        """Train the model with automatic checkpointing and resumption."""
        # Try to resume from checkpoint if requested
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint()

        if start_epoch >= self.data.tasklist.epochs:
            # last_epoch = max(0, start_epoch - 1)
            # Run one eval pass on the loaded weights to get embeddings
            _, val_outputs = self.validate_model(self.dataloader_val, "valid")
            _, test_outputs = self.validate_model(self.dataloader_test, "test")
            if self.accelerator.is_main_process and self.cfg.model.name != "logistic_regression":
                # self.save_adata_umap(test_embed, val_embed)
                # self.print_r0(f"> Saved UMAP from checkpoint epoch {last_epoch}")
                pass
            return

        # If the tracked parameter is specified
        best_metric = defaultdict(dict)
        for subtask_name, subtask in self.data.tasklist:
            if subtask.track_metric is not None:
                best_metric[subtask_name] = defaultdict(lambda: float("-inf"))
                assert (
                    subtask.track_metric in subtask.metrics
                ), "The tracking metric is not in the list of metrics, please check your configuration task file"

        # Initialize early stopping parameters
        early_stopping = self.data.tasklist.early_stopping
        early_stopping_patience = self.data.tasklist.early_stopping_patience
        patience_counter = defaultdict(int)

        def fit_epoch(epoch: int):
            # Validation and test evaluation
            valid_log, val_outputs = self.validate_model(self.dataloader_val, dataset_type="valid")
            test_log, test_outputs = self.validate_model(self.dataloader_test, dataset_type="test")

            # Track the best metric if specified
            reset_patience_counter = False
            for subtask_name, subtask in self.data.tasklist:
                if subtask.track_metric is not None:
                    val_metric = valid_log.get(f"valid_{subtask_name}_{subtask.track_metric}", float("-inf"))
                    if (
                        val_metric > best_metric[subtask_name][f"best_val_{subtask_name}_{subtask.track_metric}"]
                    ):  # Change to >= if you want to debug UMAP
                        for key in val_outputs:
                            self.best_val_outputs[key][subtask_name] = val_outputs[key][subtask_name]
                            self.best_test_outputs[key][subtask_name] = test_outputs[key][subtask_name]

                        self.best_epoch[subtask_name] = epoch

                        best_metric[subtask_name][f"best_val_{subtask_name}_{subtask.track_metric}"] = val_metric
                        self.print_r0(f"New best validation for {subtask_name} {subtask.track_metric}: {val_metric}")
                        best_metric[subtask_name]["reported_epoch"] = epoch  # log the epoch for convenience
                        for metric in subtask.metrics:
                            best_metric[subtask_name][f"reported_test_{metric}"] = test_log.get(
                                f"test_{subtask_name}_{metric}",
                                float("-inf"),
                            )

                        reset_patience_counter = True

                        # Save checkpoint for best model
                        self.save_checkpoint(epoch)
                        self.print_r0(f"> Saved best model checkpoint at epoch {epoch}")

                else:
                    for key in val_outputs:
                        self.best_val_outputs[key][subtask_name] = val_outputs[key][subtask_name]
                        self.best_test_outputs[key][subtask_name] = test_outputs[key][subtask_name]
                    self.best_epoch[subtask_name] = epoch

                if reset_patience_counter:
                    patience_counter[subtask_name] = 0  # Reset patience counter since we have a new best
                else:
                    patience_counter[subtask_name] += 1
                    if early_stopping:
                        self.print_r0(
                            f"No improvement in validation {subtask.track_metric}. "
                            f"Patience counter: {patience_counter[subtask_name]}/{early_stopping_patience}",
                        )

                # Check early stopping condition
                if early_stopping and patience_counter[subtask_name] >= early_stopping_patience:
                    self.print_r0(
                        f"Early stopping triggered. No improvement in {subtask.track_metric} for "
                        f"{early_stopping_patience} epochs.",
                    )
                    return True

            # Train for one epoch
            self.train_epoch(epoch)

            # Save checkpoint at regular intervals if requested
            if (epoch + 1) % checkpoint_every_n_epochs == 0:
                self.save_checkpoint(epoch)
                self.print_r0(f"> Saved regular checkpoint at epoch {epoch}")

            return False

        for epoch in range(start_epoch, self.data.tasklist.epochs):
            precomputation_condition = precompute_last_epoch and epoch + 1 == self.data.tasklist.epochs
            context = nullcontext()
            if precomputation_condition:
                context = PrecomputationContext(self, save_precomputed=True, get_precomputed=True, run_wandb=True)

            with context:
                stop_training = fit_epoch(epoch)
                if stop_training:
                    break

        if do_cleanup:
            if (
                self.accelerator.is_main_process
                and self.has_embeddings
                and not isinstance(self.data.datasets["full"], Heimdall.datasets.PairedInstanceDataset)
            ):
                if (
                    self.best_test_outputs
                    and self.best_val_outputs
                    and hasattr(self, "results_folder")
                    and not self.skip_umaps
                ):
                    self.save_umaps()

                    self.print_r0(f"> Saved best UMAP checkpoint at epoch {self.best_epoch}")
                else:
                    self.print_r0("> Skipped saving UMAP")

            if self.run_wandb and self.accelerator.is_main_process:
                for subtask_name, subtask in self.data.tasklist:
                    if subtask.track_metric is not None:  # logging the best val score and the tracked test scores
                        self.accelerator.log(best_metric[subtask_name])
                self.accelerator.end_training()

            if self.accelerator.is_main_process:
                self.print_r0("> Model has finished Training")

    def save_umaps(self):
        best_test_embed = self.best_test_outputs["embeddings"]
        save_umap(
            self.data,
            best_test_embed,
            split="test",
            savepath=self.results_folder,
            log_umap=self.run_wandb,
        )

        best_val_embed = self.best_val_outputs["embeddings"]
        save_umap(
            self.data,
            best_val_embed,
            split="val",
            savepath=self.results_folder,
            log_umap=self.run_wandb,
        )

    def instantiate_loss_functions_from_config(self):
        loss_functions = {}
        for subtask_name, subtask in self.data.tasklist:
            loss_kwargs = {}
            loss_name = subtask.loss_config.type.split(".")[-1]
            if loss_name.startswith("Flatten"):
                loss_kwargs["num_labels"] = self.num_labels[subtask_name]

            loss_functions[subtask_name] = instantiate_from_config(subtask.loss_config, **loss_kwargs)

        return loss_functions

    def get_precomputed_outputs(self, inputs):
        outputs = {}
        for subtask_name, _ in self.data.tasklist:
            cell_index = inputs["idx"][subtask_name].to(torch.int32).tolist()
            cls_embeddings = self.data.adata.obsm[f"{subtask_name}_cls_embeddings"][cell_index]
            cls_embeddings = torch.from_numpy(cls_embeddings).to(device=self.data.accelerator.device)

            head_output = TransformerOutput(
                logits=torch.zeros_like(cls_embeddings),
                sequence_embeddings=torch.zeros_like(cls_embeddings),
                cls_embeddings=cls_embeddings,
            )
            outputs[subtask_name] = head_output

        return outputs

    def save_precomputed_outputs(self, inputs, outputs):
        for subtask_name, _ in self.data.tasklist:
            head_output = outputs[subtask_name]
            cls_embeddings = head_output.cls_embeddings.detach().cpu().numpy()
            if f"{subtask_name}_cls_embeddings" not in self.data.adata.obsm:
                _, d_model = cls_embeddings.shape
                self.data.adata.obsm[f"{subtask_name}_cls_embeddings"] = np.zeros(
                    (self.data.adata.n_obs, *cls_embeddings.shape[1:]),
                )

            cell_index = inputs["idx"][subtask_name].to(torch.int32).tolist()
            self.data.adata.obsm[f"{subtask_name}_cls_embeddings"][cell_index] = cls_embeddings

    def get_outputs_and_loss(self, batch, cumulative_loss=None):
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

        if self.get_precomputed:
            outputs = self.get_precomputed_outputs(inputs)
        else:
            outputs = self.model(inputs=inputs)

        if self.save_precomputed:
            self.save_precomputed_outputs(inputs, outputs)

        labels = batch["labels"]
        batch_outputs = {subtask_name: {} for subtask_name, _ in self.data.tasklist}

        for subtask_name, _ in self.data.tasklist:
            if self.has_embeddings:
                batch_outputs[subtask_name]["embeddings"] = outputs[subtask_name].cls_embeddings

            batch_outputs[subtask_name]["labels"] = labels[subtask_name]

        batch_loss = {}
        if self.get_precomputed:
            for subtask_name, _ in self.data.tasklist:
                batch_outputs[subtask_name]["preds"] = torch.zeros_like(labels[subtask_name])

            return batch_outputs, batch_loss

        preds = {}
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

            batch_loss[subtask_name] = loss_function(logits, subtask_labels.clone())

        if cumulative_loss is None:
            cumulative_loss = batch_loss
        else:
            for subtask_name, _ in self.data.tasklist:
                cumulative_loss[subtask_name] += batch_loss[subtask_name]

        for subtask_name, _ in self.data.tasklist:
            batch_outputs[subtask_name]["preds"] = preds[subtask_name]

        return batch_outputs, cumulative_loss

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

            outputs = defaultdict(lambda: defaultdict(list))  # Dict of dicts

        with tqdm(dataloader, disable=not self.accelerator.is_main_process) as pbar:
            for batch in pbar:
                step += 1

                is_logging = step % log_every == 0
                lr = self.lr_scheduler.get_last_lr()[0]

                with self.accelerator.accumulate(self.model) if training else nullcontext():
                    batch_outputs, loss = self.get_outputs_and_loss(batch, loss)
                    total_loss = sum(loss.values())

                    if training:
                        self.accelerator.backward(total_loss)
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.grad_norm_clip,
                            )
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                            self.step += 1

                            pbar.set_description(
                                f"Epoch: {epoch} "
                                f"Step {self.step} "
                                f"Loss: {total_loss.item():.4f} "
                                f"LR: {lr:.1e} "
                                f"grad_norm: {grad_norm:.4f} ",
                            )

                            if is_logging:
                                log = {
                                    "train_loss": total_loss.item(),
                                    "global_step": self.step,
                                    "learning_rate": lr,
                                    "epoch": epoch,
                                    "grad_norm": grad_norm,
                                }
                                if self.run_wandb and self.accelerator.is_main_process:
                                    self.accelerator.log(log)

                            with torch.no_grad():
                                for param in constrained_params:
                                    # TODO: make this more robust to different types of PGD
                                    project2simplex_(param, dim=0)

                        loss = None
                    else:
                        for subtask_name, _ in self.data.tasklist:
                            for key, value in batch_outputs[subtask_name].items():
                                outputs[key][subtask_name].extend(value.detach().cpu().numpy())

                    if metrics is not None and not self.get_precomputed:
                        for subtask_name, subtask in self.data.tasklist:
                            for metric_name, metric in metrics[subtask_name].items():  # noqa: B007
                                # Built-in metric
                                subtask_labels = batch_outputs[subtask_name]["labels"]
                                subtask_preds = batch_outputs[subtask_name]["preds"]

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

                if self.fastdev:
                    break

            if not training:
                for key in outputs:
                    for subtask_name, _ in self.data.tasklist:
                        outputs[key][subtask_name] = np.array(outputs[key][subtask_name])

        return outputs, loss

    def validate_model(self, dataloader, dataset_type):
        self.model.eval()
        metrics = self._initialize_metrics()

        if len(dataloader) == 0:
            raise ValueError("`DataLoader` length cannot be zero. Check custom sampler implementation.")

        with torch.no_grad():
            outputs, dataloader_loss = self.iterate_dataloader(
                dataloader,
                metrics=metrics,
            )

        loss = {subtask_name: subtask_loss / len(dataloader) for subtask_name, subtask_loss in dataloader_loss.items()}
        total_loss = sum(loss.values())

        if self.accelerator.num_processes > 1:
            loss_tensor = torch.tensor(
                [total_loss],
                device=self.accelerator.device,
            )
            # loss is a python floating point value, for gather
            # operation across multiple processes needs to be
            # cuda tensor
            total_loss = self.accelerator.gather(loss_tensor).mean().item()

        log = {f"{dataset_type}_{subtask_name}_loss": subtask_loss for subtask_name, subtask_loss in loss.items()}
        log[f"{dataset_type}_loss"] = total_loss

        if self.save_precomputed:
            return log, outputs

        for subtask_name, subtask in self.data.tasklist:
            for metric_name, metric in metrics[subtask_name].items():
                if metric_name != "ConfusionMatrix":
                    # Built-in metric
                    log[f"{dataset_type}_{subtask_name}_{metric_name}"] = metric.compute().item()
                    if metric_name.startswith(("Accuracy", "Precision", "Recall", "F1Score")):
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
                    self.accelerator.log({f"{dataset_type}_{subtask_name}_topk_acc_curve": chart})

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
                    name: float(acc) for name, acc in zip(self.class_names[subtask_name], per_class_acc)
                }

                # 4. Log interactive confusion matrix to WandB (main process only)
                if self.run_wandb and self.accelerator.is_main_process:
                    y_true_np = outputs["labels"][subtask_name]
                    y_pred_np = outputs["preds"][subtask_name]

                    # Convert logits/probs to hard labels if needed
                    if y_pred_np.ndim > 1:  # shape (N, C)
                        y_pred_np = y_pred_np.argmax(axis=1)

                    # Flatten & convert to Python lists
                    y_true_list = y_true_np.reshape(-1).tolist()
                    y_pred_list = y_pred_np.reshape(-1).tolist()

                    wandb_cm = wandb.plot.confusion_matrix(
                        y_true=y_true_list,
                        preds=y_pred_list,
                        class_names=self.class_names[subtask_name],  # same order as metric
                    )
                    self.accelerator.log(
                        {f"{dataset_type}_{subtask_name}_confusion_matrix": wandb_cm},
                    )

        rss = self.process.memory_info().rss / (1024**3)
        log["Process_mem_rss"] = rss

        if self.run_wandb and self.accelerator.is_main_process:
            self.accelerator.log(log)

        if not self.run_wandb and self.accelerator.is_main_process:
            print(f"{dataset_type}_log = {pformat(log)}")

        return log, outputs

    def train_epoch(self, epoch):
        self.model.train()

        self.iterate_dataloader(
            self.dataloader_train,
            epoch=epoch,
        )

    def initialize_checkpointing(self, additional_keys: tuple = (), hash_vars: tuple = ()):
        """Initialize checkpoint directory."""
        if getattr(self.cfg, "work_dir", None) is not None:
            self.results_folder = Path(self.cfg.work_dir)
        else:
            cache_dir = self.cfg.cache_preprocessed_dataset_dir
            keys = self.CHECKPOINT_KEYS + additional_keys
            self.results_folder, _, _ = get_fully_qualified_cache_paths(
                self.cfg,
                cache_dir,
                keys=keys,
                hash_vars=hash_vars,
            )

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
        if not hasattr(self, "results_folder") or not self.results_folder.exists():
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
        if not hasattr(self, "results_folder") or not self.results_folder.exists():
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

        epoch = self.load_trainer_state(data)

        if "version" in data:
            self.print_r0(f"> Checkpoint version: {data['version']}")
        self.print_r0(f"> Resumed from epoch {epoch}, step {self.step}")

        return epoch + 1

    def load_trainer_state(self, data):
        # Restore optimizer and scheduler states

        clean_opt_sd = clean_optimizer_state_for_current_model(data["optimizer"], self.optimizer, verbose=True)

        self.optimizer.load_state_dict(clean_opt_sd)
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

        return epoch

    def load_pretrained(self):
        self.initialize_checkpointing()

        # Load the checkpoint
        config = self.cfg
        if "pretrained_milestone" in config:
            load_path = self.results_folder / f"model-{config.pretrained_milestone}.pt"
            if not load_path.exists():
                self.print_r0(
                    f"> Checkpoint file {load_path} does not exist. `{config.pretrained_milestone=}` is invalid.",
                )
                return
        elif "pretrained_ckpt_path" in config:
            load_path = Path(config.pretrained_ckpt_path)
            if not load_path.exists():
                self.print_r0(
                    f"> Checkpoint file {load_path} does not exist. Check the value of "
                    f"`{config.pretrained_ckpt_path=}` for correctness.",
                )
                return
        else:
            return

        self.print_r0(f"> Loading pretrained model state from {load_path}")

        # Load the data
        device = self.accelerator.device
        data = torch.load(str(load_path), map_location=device, weights_only=False)

        pretrained_state_dict = data["model"]

        filtered_pretrained_params = OrderedDict(
            filter(lambda param_tuple: "decoder" not in param_tuple[0], pretrained_state_dict.items()),
        )  # we drop the pretrained head and load all other params

        # Unwrap model and restore parameters
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(filtered_pretrained_params, strict=False)
        # model.load_state_dict(data["model"])

        self.load_trainer_state(data)

        if self.accelerator.is_main_process:
            print(f">Finished loading pretrained params loaded from {load_path}")


def setup_trainer_generic(config, setup_model: Callable, cpu=True):
    accelerator, cr, run_wandb, only_preprocess_data = setup_data(config)

    if only_preprocess_data:
        return

    model = setup_model(config, cr, is_main_process=accelerator.is_main_process)
    trainer = instantiate_from_config(
        config.trainer,
        cfg=config,
        model=model,
        data=cr,
        accelerator=accelerator,
        run_wandb=run_wandb,
    )
    trainer.load_pretrained()

    return trainer


def setup_trainer(config, cpu=True):
    return setup_trainer_generic(config, setup_model=setup_model, cpu=cpu)


class PrecomputationContext:
    ATTRIBUTES = ("save_precomputed", "get_precomputed", "run_wandb")

    def __init__(
        self,
        trainer: HeimdallTrainer,
        save_precomputed: bool,
        get_precomputed: bool,
        run_wandb: bool = False,
    ):
        self.trainer = trainer
        self.save_precomputed = save_precomputed
        self.get_precomputed = get_precomputed
        self.run_wandb = run_wandb

    def swap(self):
        for attribute in self.ATTRIBUTES:
            context_attr = getattr(self, attribute)
            trainer_attr = getattr(self.trainer, attribute)
            setattr(self.trainer, attribute, context_attr)
            setattr(self, attribute, trainer_attr)

    def __enter__(self):
        self.swap()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if self.trainer.save_precomputed and isinstance(trainer, PartitionedCellRepresentation):
        #     self.trainer.data.partition = None

        self.swap()

        return False


from copy import deepcopy

import torch


def clean_optimizer_state_for_current_model(saved_opt_sd: dict, optimizer: torch.optim.Optimizer, verbose: bool = True):
    """Return a cleaned optimizer state_dict compatible with the given
    `optimizer`.

    - saved_opt_sd: the checkpoint["optimizer"] you loaded from disk
    - optimizer: the optimizer instance that was created for the current model (and whose .param_groups define grouping)

    """

    saved_state = deepcopy(saved_opt_sd.get("state", {}))
    saved_param_group_list = saved_opt_sd.get("param_groups", [])
    if verbose:
        print(
            f"[clean_optimizer] saved state entries: {len(saved_state)}, saved param_groups: {len(saved_param_group_list)}",
        )

    # Build a list of the current parameters in the order of optimizer.param_groups
    current_param_groups = optimizer.param_groups  # these have 'params' as param objects
    current_params_flat = []
    for g in current_param_groups:
        for p in g["params"]:
            current_params_flat.append(p)

    # Prepare containers for new state and param_groups
    new_state = {}
    new_param_groups = []

    # We'll mark which saved pids have been used (so we don't reuse them)
    available_saved_pids = set(saved_state.keys())

    # Helper to get a representative tensor from saved-state entry for shape check
    def representative_tensor_from_state_entry(state_entry):
        # typical keys: 'exp_avg', 'exp_avg_sq' (or for some optimizers 'momentum_buffer')
        for v in state_entry.values():
            if isinstance(v, torch.Tensor):
                return v
        return None

    matched = 0
    dropped = 0

    # For each current param group, create a new group copying hyperparams but setting 'params' to ids
    for group in current_param_groups:
        # copy group hyperparams except params
        new_group = {k: deepcopy(v) for k, v in group.items() if k != "params"}
        new_group_param_ids = []

        for param in group["params"]:
            p_shape = param.shape
            match_pid = None
            match_state = None

            # Find a saved PID with a representative tensor that matches the shape
            for saved_pid in list(available_saved_pids):
                state_entry = saved_state[saved_pid]
                rep = representative_tensor_from_state_entry(state_entry)
                if rep is None:
                    # if no tensor in state entry, we can't compare shapes; skip
                    continue
                if tuple(rep.shape) == tuple(p_shape):
                    match_pid = saved_pid
                    match_state = state_entry
                    break

            if match_pid is not None:
                # Assign saved state to this current parameter id
                new_pid = id(param)
                new_state[new_pid] = match_state
                available_saved_pids.remove(match_pid)
                new_group_param_ids.append(new_pid)
                matched += 1
            else:
                # No matching saved state for this param (likely a newly initialized param)
                new_group_param_ids.append(id(param))
                dropped += 1

        new_group["params"] = new_group_param_ids
        new_param_groups.append(new_group)

    if verbose:
        print(f"[clean_optimizer] matched saved states -> {matched}")
        print(f"[clean_optimizer] params without saved state (new) -> {dropped}")
        print(f"[clean_optimizer] leftover saved states not matched -> {len(available_saved_pids)}")

    cleaned_sd = {"state": new_state, "param_groups": new_param_groups}
    # Copy over (safe) additional keys if present (like 'defaults') from the saved dict,
    # but param_groups is the critical part that must match the current optimizer.
    if "defaults" in saved_opt_sd:
        cleaned_sd["defaults"] = deepcopy(saved_opt_sd["defaults"])

    return cleaned_sd
