"""Heimdall trainer."""

import psutil
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score, MatthewsCorrCoef, Precision, Recall
from torchmetrics.regression import MeanSquaredError, R2Score
from tqdm import tqdm
from transformers import get_scheduler

import Heimdall.losses


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

    def fit(self):
        """This is the main trainer.fit() function that is called for
        training."""

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

        for epoch in range(self.cfg.tasks.args.epochs):
            # Validation and test evaluation
            valid_log = self.validate_model(self.dataloader_val, dataset_type="valid")
            test_log = self.validate_model(self.dataloader_test, dataset_type="test")

            # Track the best metric if specified
            if track_metric:
                val_metric = valid_log.get(f"valid_{track_metric}", float("-inf"))
                if val_metric > best_metric[f"best_val_{track_metric}"]:
                    best_metric[f"best_val_{track_metric}"] = val_metric
                    print(f"New best validation {track_metric}: {val_metric}")
                    best_metric["reported_epoch"] = epoch  # log the epoch for convenience
                    for metric in self.cfg.tasks.args.metrics:
                        best_metric[f"reported_test_{metric}"] = test_log.get(f"test_{metric}", float("-inf"))
                        # print(f"Corresponding test {metric}: {best_metric[f'reported_test_{metric}']}")
                    patience_counter = 0  # Reset patience counter since we have a new best
                else:
                    patience_counter += 1
                    if early_stopping:
                        print(
                            f"No improvement in validation {track_metric}. "
                            f"Patience counter: {patience_counter}/{early_stopping_patience}",
                        )

            # Check early stopping condition
            if early_stopping and patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered. No improvement in {track_metric} for {early_stopping_patience} epochs.",
                )
                break

            # Training for the current epoch
            self.train_epoch(epoch)

        if self.run_wandb and self.accelerator.is_main_process:
            if track_metric:  # logging the best val score and the tracked test scores
                self.accelerator.log(best_metric)

            self.accelerator.end_training()

        if self.accelerator.is_main_process:
            print("> Model has finished Training")

    def get_loss(self, logits, labels, masks=None):
        if masks is not None:
            logits, labels = logits[masks], labels[masks]

        if self.custom_loss_func:
            loss = self.loss_fn(logits, labels)
        elif self.cfg.loss.name.endswith("BCEWithLogitsLoss"):
            loss = self.loss_fn(logits, labels)
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
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_norm_clip)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                t.set_description(f"Epoch: {epoch}, Step {step}, Loss: {loss.item():.4f}, LR: {lr:.1e}")

                if is_logging:
                    log = {
                        "train_loss": loss.item(),
                        "step": step,
                        "learning_rate": lr,
                        "epoch": epoch,
                    }
                    if self.run_wandb and self.accelerator.is_main_process:
                        self.accelerator.log(log)

                if self.cfg.trainer.fastdev:
                    break

    def validate_model(self, dataloader, dataset_type):
        self.model.eval()
        metrics = self._initialize_metrics()
        # print(metrics)
        loss = 0

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

                    # breakpoint()
                    metric.update(logits, labels)
                    # if callable(metric):
                    #     # Custom metric
                    #     print("entered custom")
                    #     metric_value = metric(logits, labels)
                    #     if isinstance(metric_value, torch.Tensor):
                    #         metric_value = metric_value.item()
                    #     metrics[metric_name] = metric_value
                    # else:
                    #     # Built-in metric
                    #     print(metric)
                    #     print(metric_name)
                    #     metric.update(logits, labels)

                if self.cfg.trainer.fastdev:
                    break

        loss = loss / len(dataloader)
        if self.accelerator.num_processes > 1:
            loss = self.accelerator.gather(torch.tensor(loss)).mean().item()

        log = {f"{dataset_type}_loss": loss}
        for metric_name, metric in metrics.items():
            if metric_name != "ConfusionMatrix":
                # Built-in metric
                log[f"{dataset_type}_{metric_name}"] = metric.compute().item()
                if metric_name in ["Accuracy", "Precision", "Recall", "F1Score", "MathewsCorrCoef"]:
                    log[f"{dataset_type}_{metric_name}"] *= 100  # Convert to percentage for these metrics

        if "ConfusionMatrix" in metrics and not callable(metrics["ConfusionMatrix"]):
            confusion_matrix = metrics["ConfusionMatrix"].compute()
            per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
            per_class_acc = per_class_acc.cpu().numpy() * 100
            log[f"{dataset_type}_per_class_accuracy"] = {f"class_{i}": acc for i, acc in enumerate(per_class_acc)}

        rss = self.process.memory_info().rss / (1024**3)
        log["Process_mem_rss"] = rss

        if self.run_wandb and self.accelerator.is_main_process:
            self.accelerator.log(log)

        if not self.run_wandb and self.accelerator.is_main_process:
            print(log)

        return log
