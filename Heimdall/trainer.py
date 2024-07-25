"""Heimdall trainer."""

import psutil
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score, MatthewsCorrCoef, Precision, Recall
from tqdm import tqdm
from transformers import get_scheduler


class HeimdallTrainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        run_wandb=False,
    ):
        """Initialize the trainer.

        Args:
            config (dict): Configuration dictionary.

        """

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.dataloader_test = dataloader_test
        self.run_wandb = run_wandb
        self.process = psutil.Process()  # tracking the RSS metrics of this experiment

        # set up accelerator DDP
        accelerator_log_kwargs = {}
        if run_wandb:
            accelerator_log_kwargs["log_with"] = "wandb"
            accelerator_log_kwargs["project_dir"] = config.work_dir

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.trainer.accumulate_grad_batches,
            step_scheduler_with_optimizer=False,
            **accelerator_log_kwargs,
        )
        set_seed(config.seed)
        self.accelerator.wait_for_everyone()
        print(f"> Using Device: {self.accelerator.device}")

        # Initialize W&B
        # init wandb tracking
        if run_wandb is True and self.accelerator.is_main_process:
            print("==> Starting a new WANDB run")
            new_tags = (config.dataset.dataset_name, config.f_g.name, config.f_c.name)
            wandb_config = {
                "wandb": {
                    "tags": new_tags,
                    "name": config.run_name,
                    "entity": config.entity,
                },
            }

            env_config = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
            self.accelerator.init_trackers(
                project_name=config.project_name,
                config=env_config,
                init_kwargs=wandb_config,
            )
            print("==> Initialized Run")

        torch.cuda.empty_cache()

        # cosine LR scheduler
        dataset_config = config.dataset.task_args
        global_batch_size = dataset_config["batchsize"]
        total_steps = len(dataloader_train.dataset) // global_batch_size * dataset_config["epochs"]
        warmup_ratio = config.scheduler.warmup_ratio
        warmup_step = int(warmup_ratio * total_steps)

        if self.accelerator.is_main_process:
            print(" !!! Remember that config batchsize here is GLOBAL Batchsize !!!")
            print(f"> global batchsize: {dataset_config['batchsize']}")
            print(f"> num_devices: {self.accelerator.num_processes}")
            print(f"> total_samples: {len(dataloader_train.dataset)}")
            print(f"> warmup_step: {warmup_step}")
            print(f"> total_steps: {total_steps}")
            print(f"> per_device_batch_size: {dataset_config['batchsize'] // self.accelerator.num_processes}")

        self.lr_scheduler = get_scheduler(
            name=self.config.scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=total_steps,
        )

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

    def train(self):

        for epoch in range(self.config.dataset.task_args.epochs):
            # val first so we can see the randomized performance before the first epoch
            self.validate_model(self.dataloader_val, dataset_type="valid")
            # test first so we can see the randomized performance before the first epoch
            self.validate_model(self.dataloader_test, dataset_type="test")
            self.train_epoch(epoch)

        if self.run_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()

        if self.accelerator.is_main_process:
            print("> Model has finished Training")

    def train_epoch(self, epoch):
        accelerator = self.accelerator
        model = self.model
        optimizer = self.optimizer
        config = self.config
        dataloader_train = self.dataloader_train
        run_wandb = self.run_wandb
        lr_scheduler = self.lr_scheduler

        step = len(dataloader_train) * epoch
        log_every = 1  # this is the logging frequency, default set to 1 because wandb seems to not be the bottleneck
        model.train()
        # for i, batch in enumerate(tqdm(dataloader_train, disable=not accelerator.is_main_process)):
        with tqdm(dataloader_train, disable=not accelerator.is_main_process) as t:
            for batch in t:
                # t0 = time.time()
                step += 1
                is_logging = step % log_every == 0

                lr = lr_scheduler.get_lr()[0]
                with accelerator.accumulate(model):

                    if len(batch["conditional_tokens"]) > 0:
                        outputs = model(
                            inputs=batch["inputs"],
                            labels=batch["labels"],
                            conditional_tokens=batch["conditional_tokens"],
                        )
                    else:
                        outputs = model(inputs=batch["inputs"], labels=batch["labels"])

                    loss = outputs["loss"]
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config.optimizer.grad_norm_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                t.set_description(f"Epoch: {epoch}, Step {step}, Loss: {loss.item():.4f}, LR: {lr:.1e}")

                if is_logging:
                    log = {
                        "train_loss": loss.item(),
                        "step": step,
                        "learning_rate": lr,
                    }

                    if run_wandb and accelerator.is_main_process:
                        accelerator.log(log)

                    # if not run_wandb and accelerator.is_main_process:
                #     print(log)

    def validate_model(self, dataloader_val, dataset_type):
        accelerator = self.accelerator
        model = self.model
        config = self.config
        run_wandb = self.run_wandb

        num_classes = config.dataset.task_args.prediction_dim

        # Accuracy, Precision, Recall, F1, Matthew's Correlation
        acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(accelerator.device)
        precision_metric = Precision(task="multiclass", num_classes=num_classes, average="macro").to(accelerator.device)
        recall_metric = Recall(task="multiclass", num_classes=num_classes, average="macro").to(accelerator.device)
        f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(accelerator.device)
        mcc_metric = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(accelerator.device)
        # For micro average, set average='micro' in Precision and Recall
        precision_micro = Precision(task="multiclass", num_classes=num_classes, average="micro").to(accelerator.device)
        recall_micro = Recall(task="multiclass", num_classes=num_classes, average="micro").to(accelerator.device)
        # per class accuracy using confusion matrix
        confusion_matrix_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(accelerator.device)

        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader_val, disable=not accelerator.is_main_process):
                if len(batch["conditional_tokens"]) > 0:
                    outputs = model(
                        inputs=batch["inputs"],
                        labels=batch["labels"],
                        conditional_tokens=batch["conditional_tokens"],
                    )
                else:
                    outputs = model(inputs=batch["inputs"], labels=batch["labels"])

                logits = outputs["logits"]
                reshaped_logits = logits.reshape((-1, logits.size(-1)))
                reshaped_labels = batch["labels"].reshape(-1)
                loss += outputs["loss"].item()
                # accuracy, precision, recall, f1, precision_micro, recall_micro, class-wise accuracy
                acc_metric.update(reshaped_logits, reshaped_labels)
                precision_metric.update(reshaped_logits, reshaped_labels)
                recall_metric.update(reshaped_logits, reshaped_labels)
                f1_metric.update(reshaped_logits, reshaped_labels)
                mcc_metric.update(reshaped_logits, reshaped_labels)
                precision_micro.update(reshaped_logits, reshaped_labels)
                recall_micro.update(reshaped_logits, reshaped_labels)
                confusion_matrix_metric.update(reshaped_logits, reshaped_labels)
        try:
            loss = accelerator.gather(loss).sum() / (len(dataloader_val))
        except:  # FIX: E722 do not use bare 'except'
            loss = loss / (len(dataloader_val))
        acc = acc_metric.compute() * 100
        precision = precision_metric.compute() * 100
        recall = recall_metric.compute() * 100
        f1 = f1_metric.compute() * 100
        mcc = mcc_metric.compute() * 100  # Convert to percentage
        precision_micro_value = precision_micro.compute() * 100
        recall_micro_value = recall_micro.compute() * 100

        # Calculating per-class accuracy
        confusion_matrix = confusion_matrix_metric.compute()
        per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        per_class_acc = per_class_acc.cpu().numpy() * 100  # Convert to numpy array and scale to percentage
        rss = self.process.memory_info().rss / (1024**3)  # convert to gigabytes

        class_wise_acc_dict = {f"class_{i}": acc for i, acc in enumerate(per_class_acc)}

        log = {
            f"{dataset_type}_loss": loss,
            f"{dataset_type}_acc": acc,
            f"{dataset_type}_precision": precision,
            f"{dataset_type}_recall": recall,
            f"{dataset_type}_f1": f1,
            f"{dataset_type}_mcc": mcc,
            f"{dataset_type}_precision_micro": precision_micro_value,
            f"{dataset_type}_recall_micro": recall_micro_value,
            f"{dataset_type}_per_class_accuracy": class_wise_acc_dict,
            "Process_mem_rss": rss,
        }

        if run_wandb and accelerator.is_main_process:
            accelerator.log(log)

        if not run_wandb and accelerator.is_main_process:
            print(log)

        return mcc
