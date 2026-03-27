import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor
from pytorch_lightning.loggers.wandb import WandbLogger

# ------------------------------------------------------------------
# These helpers appear to provide:
# - TorchMetricClassification / TorchMetricRegression:
#     metric collections used during training / validation / testing
# - wandb_* curve helpers:
#     convenience functions for ROC / PR / DET logging to Weights & Biases
# ------------------------------------------------------------------
from src.models.eval import (
    TorchMetricClassification,
    TorchMetricRegression,
    wandb_detection_error_tradeoff_curve,
    wandb_pr_curve,
    wandb_roc_curve,
)

# ------------------------------------------------------------------
# Utility helpers:
# - binary_logits_to_pos_probs:
#     converts raw classifier logits into positive-class probabilities
# - upload_pandas_df_to_wandb:
#     uploads a DataFrame (e.g. test predictions) into a wandb table
# ------------------------------------------------------------------
from src.utils import (
    binary_logits_to_pos_probs,
    get_logger,
    upload_pandas_df_to_wandb,
)

logger = get_logger(__name__)


class SensingModel(pl.LightningModule):
    """
    Base Lightning model for all trainable sensing models in this project.

    The main idea:
    - subclasses define the actual architecture and forward()
    - this base class provides the standard training / validation / test loops
    - this class also manages metric tracking, wandb logging, and bookkeeping

    Important expectation for subclasses:
    ------------------------------------
    forward(x, y) should return:
        loss, preds

    where:
    - loss is a scalar tensor
    - preds are the raw model outputs (usually logits for classification)
    """

    def __init__(
        self,
        metric_class: torchmetrics.MetricCollection,
        learning_rate: float = 1e-3,
        val_bootstraps: int = 100,
        warmup_steps: int = 0,
        batch_size: int = 800,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Parameters
        ----------
        metric_class:
            A metric collection constructor, e.g.
            TorchMetricClassification or TorchMetricRegression.

            This is one of the key things determined by the wrapper classes
            ClassificationModel / RegressionModel below.

        learning_rate:
            Base optimizer learning rate.

        val_bootstraps:
            Number of bootstrap samples used for validation/test metrics
            if the metric collection supports bootstrapped confidence intervals.

        warmup_steps:
            Linear learning-rate warmup duration in optimizer steps.

        batch_size:
            Stored mainly for bookkeeping / hyperparameter logging.

        input_shape:
            Optional shape metadata (often used by subclasses that need
            to know sequence length / feature count when building modules).
        """
        super(SensingModel, self).__init__()

        # --------------------------------------------------------------
        # Per-epoch caches.
        #
        # These are used mainly for:
        # - epoch-end metric computation
        # - logging ROC/PR/DET curves
        # - storing full test predictions for later export
        #
        # Note:
        # There is some duplication in the original initialization
        # (e.g. train_labels / val_preds assigned twice), but functionally
        # it just means these start as empty lists.
        # --------------------------------------------------------------
        self.val_preds = []
        self.train_labels = []

        self.train_preds = []
        self.train_labels = []

        self.val_preds = []
        self.val_labels = []

        self.test_preds = []
        self.test_labels = []
        self.test_participant_ids = []
        self.test_dates = []
        self.test_losses = []

        # Optional references to datasets if subclasses or outer code
        # decide to attach them.
        self.train_dataset = None
        self.eval_dataset = None

        # Number of bootstrap resamples used in val/test metrics.
        self.num_val_bootstraps = val_bootstraps

        # --------------------------------------------------------------
        # Metric collections for each split.
        #
        # Why three copies?
        # Because train/val/test are tracked independently and often
        # need their own internal running state.
        #
        # The metric_class itself is chosen by the model type:
        # - classification models use TorchMetricClassification
        # - regression models use TorchMetricRegression
        # --------------------------------------------------------------
        self.train_metrics = metric_class(bootstrap_samples=0, prefix="train/")
        self.val_metrics = metric_class(
            bootstrap_samples=self.num_val_bootstraps, prefix="val/"
        )
        self.test_metrics = metric_class(
            bootstrap_samples=self.num_val_bootstraps, prefix="test/"
        )

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size

        # Useful metadata fields often filled later.
        self.wandb_id = None
        self.name = None

        # Save constructor args into Lightning hyperparameters.
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parser):
        """
        Adds generic optimization/training arguments to an argparse parser.

        This is not architecture-specific; it is shared across models
        that subclass SensingModel.
        """
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            help="Base learning rate",
        )
        parser.add_argument(
            "--warmup_steps",
            type=int,
            default=0,
            help="Steps until the learning rate reaches its maximum values",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=800,
            help="Training batch size",
        )
        parser.add_argument(
            "--num_val_bootstraps",
            type=int,
            default=100,
            help="Number of bootstraps to use for validation metrics. Set to 0 to disable bootstrapping.",
        )
        return parser

    def on_train_start(self) -> None:
        """
        Ensures the metric objects are moved to the same device as the model
        at the start of training.

        This matters because metric states may themselves be tensors.
        """
        self.train_metrics.apply(lambda x: x.to(self.device))
        self.val_metrics.apply(lambda x: x.to(self.device))
        return super().on_train_start()

    def training_step(
        self, batch, batch_idx
    ) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        """
        Standard Lightning training step.

        Expected batch format:
            batch["inputs_embeds"] -> model input tensor
            batch["label"]         -> target tensor

        Expected subclass behavior:
            self.forward(x, y) returns (loss, preds)
        """
        # Move / cast inputs.
        # This code explicitly casts to CUDA float tensor, which assumes
        # training happens on GPU. If CPU support is needed, this is one
        # place that would need softening.
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]

        # Delegate actual model/loss computation to subclass forward().
        loss, preds = self.forward(x, y)

        # Log step-level training loss.
        self.log("train/loss", loss.item(), on_step=True)

        # Detach predictions before metric bookkeeping so we do not
        # retain the whole computation graph.
        preds = preds.detach()
        y = y.detach()

        # Update running training metrics.
        self.train_metrics.update(preds, y)

        # --------------------------------------------------------------
        # Only classifiers store full per-batch predictions/labels for
        # epoch-end ROC/PR/DET plotting.
        #
        # Regression models do not use those curve plots.
        # --------------------------------------------------------------
        if self.is_classifier:
            self.train_preds.append(preds.detach().cpu())
            self.train_labels.append(y.detach().cpu())

        return {"loss": loss, "preds": preds, "labels": y}

    def on_train_epoch_end(self):
        """
        End-of-epoch hook for training.

        What happens here:
        1. For classifiers with wandb enabled, build ROC/PR/DET curves
           from all cached training predictions.
        2. Compute aggregate training metrics from self.train_metrics.
        3. Log those metrics.
        4. Reset caches and metric state for next epoch.
        """
        # --------------------------------------------------------------
        # In distributed training, only rank 0 should log heavyweight
        # wandb plots. Other ranks may have a dummy logger.
        # --------------------------------------------------------------
        if (
            os.environ.get("LOCAL_RANK", "0") == "0"
            and self.is_classifier
            and isinstance(self.logger, WandbLogger)
        ):
            train_preds = torch.cat(self.train_preds, dim=0)
            train_labels = torch.cat(self.train_labels, dim=0)

            self.logger.experiment.log(
                {"train/roc": wandb_roc_curve(train_preds, train_labels, limit=9999)},
                commit=False,
            )
            self.logger.experiment.log(
                {"train/pr": wandb_pr_curve(train_preds, train_labels)},
                commit=False,
            )
            self.logger.experiment.log(
                {
                    "train/det": wandb_detection_error_tradeoff_curve(
                        train_preds, train_labels, limit=9999
                    )
                },
                commit=False,
            )

        # Aggregate metrics over the full epoch.
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True)

        # Reset state for next epoch.
        self.train_metrics.reset()
        self.train_preds = []
        self.train_labels = []

        super().on_train_epoch_end()

    def on_train_epoch_start(self):
        """
        Makes sure training metrics live on the correct device when a new epoch starts.
        """
        self.train_metrics.to(self.device)

    def on_test_epoch_end(self):
        """
        End-of-test hook.

        What happens here:
        1. Concatenate all stored test predictions / labels / metadata
        2. Optionally log classifier curves to wandb
        3. Compute and log test metrics
        4. Build a predictions DataFrame for later export
        5. Reset all test caches

        This is especially useful because test batches also store:
        - participant ids
        - dates
        so downstream analysis can be linked back to individuals/windows.
        """
        test_preds = torch.cat(self.test_preds, dim=0)
        test_labels = torch.cat(self.test_labels, dim=0)
        test_dates = np.concatenate(self.test_dates, axis=0)
        test_participant_ids = np.concatenate(self.test_participant_ids, axis=0)

        if (
            os.environ.get("LOCAL_RANK", "0") == "0"
            and self.is_classifier
            and isinstance(self.logger, WandbLogger)
        ):
            self.logger.experiment.log(
                {"test/roc": wandb_roc_curve(test_preds, test_labels, limit=9999)},
                commit=False,
            )
            self.logger.experiment.log(
                {"test/pr": wandb_pr_curve(test_preds, test_labels)},
                commit=False,
            )
            self.logger.experiment.log(
                {
                    "test/det": wandb_detection_error_tradeoff_curve(
                        test_preds, test_labels, limit=9999
                    )
                },
                commit=False,
            )

        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        # --------------------------------------------------------------
        # For binary classifiers, convert logits to positive-class probs
        # and build a DataFrame with participant/date/label/pred.
        #
        # This is extremely useful for later debugging and error analysis.
        # --------------------------------------------------------------
        pos_probs = binary_logits_to_pos_probs(test_preds.cpu().numpy())
        self.predictions_df = pd.DataFrame(
            zip(
                test_participant_ids,
                test_dates,
                test_labels.cpu().numpy(),
                pos_probs,
            ),
            columns=["participant_id", "date", "label", "pred"],
        )

        # Reset test state.
        self.test_metrics.reset()
        self.test_preds = []
        self.test_labels = []
        self.test_participant_ids = []
        self.test_dates = []

        # Note: the original code calls on_validation_epoch_end() here,
        # which is a bit unusual and may just be a typo in the project.
        super().on_validation_epoch_end()

    def predict_step(self, batch: Any) -> Any:
        """
        Lightning predict step.

        Unlike test_step, this is typically used when the user wants
        raw outputs for downstream analysis or export.

        Here it returns:
        - loss
        - logits
        - labels
        - participant id
        - end date

        Notes:
        - `probs` is computed here but not returned in the original code.
        - For binary classification, logits are often more useful than
          probabilities because thresholds can be applied later.
        """
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]

        with torch.no_grad():
            loss, logits = self.forward(x, y)
            probs = torch.nn.functional.softmax(logits, dim=1)[:, -1]

        return {
            "loss": loss,
            "preds": logits,
            "labels": y,
            "participant_id": batch["participant_id"],
            "end_date": batch["end_date_str"],
        }

    def test_step(
        self, batch, batch_idx
    ) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        """
        Standard test step.

        Similar to validation_step, but additionally stores:
        - participant ids
        - end dates

        so epoch-end code can produce a predictions table.
        """
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]
        dates = batch["end_date_str"]
        participant_ids = batch["participant_id"]

        loss, preds = self.forward(x, y)

        self.log("test/loss", loss.item(), on_step=True, sync_dist=True)

        self.test_preds.append(preds.detach())
        self.test_labels.append(y.detach())
        self.test_participant_ids.append(participant_ids)
        self.test_dates.append(dates)

        self.test_metrics.update(preds, y)
        return {"loss": loss, "preds": preds, "labels": y}

    def on_validation_epoch_end(self):
        """
        End-of-validation hook.

        For classifiers:
        - log ROC / PR / DET curves to wandb

        For all model types:
        - compute validation metrics
        - log them
        - reset val metric state and cached predictions
        """
        if (
            os.environ.get("LOCAL_RANK", "0") == "0"
            and self.is_classifier
            and isinstance(self.logger, WandbLogger)
        ):
            val_preds = torch.cat(self.val_preds, dim=0)
            val_labels = torch.cat(self.val_labels, dim=0)

            self.logger.experiment.log(
                {"val/roc": wandb_roc_curve(val_preds, val_labels, limit=9999)},
                commit=False,
            )
            self.logger.experiment.log(
                {"val/pr": wandb_pr_curve(val_preds, val_labels)},
                commit=False,
            )
            self.logger.experiment.log(
                {
                    "val/det": wandb_detection_error_tradeoff_curve(
                        val_preds, val_labels, limit=9999
                    )
                },
                commit=False,
            )

        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        self.val_metrics.reset()
        self.val_preds = []
        self.val_labels = []

        super().on_validation_epoch_end()

    def validation_step(
        self, batch, batch_idx
    ) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        """
        Standard validation step.

        Runs forward pass, logs validation loss, updates running metrics,
        and for classifiers caches logits/labels for epoch-end plotting.
        """
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]

        loss, preds = self.forward(x, y)

        self.log("val/loss", loss.item(), on_step=True, sync_dist=True)

        if self.is_classifier:
            self.val_preds.append(preds.detach())
            self.val_labels.append(y.detach())

        self.val_metrics.update(preds, y)
        return {"loss": loss, "preds": preds, "labels": y}

    def configure_optimizers(self):
        """
        Defines the optimizer and scheduler.

        Current behavior:
        - optimizer: Adam
        - scheduler: LambdaLR implementing a simple linear warmup

        The scheduler returns a multiplier:
            min(1, (step+1)/warmup_steps)

        so the LR ramps up linearly until warmup_steps, then stays flat.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def scheduler(step):
            return min(1.0, float(step + 1) / max(self.warmup_steps, 1))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler)

        return [optimizer], [
            {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "reduce_on_plateau": False,
                "monitor": "val_loss",
            }
        ]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Custom optimizer step hook.

        Right now it simply calls optimizer.step(closure=...).
        This exists mostly because Lightning allows users to customize
        stepping behavior here.
        """
        optimizer.step(closure=optimizer_closure)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Makes checkpoint loading more forgiving when architecture changes.

        What it does:
        - if a parameter exists in checkpoint and current model but shapes differ:
            skip loading that parameter
        - if a checkpoint parameter does not exist in current model:
            drop it
        - if anything changed:
            drop optimizer state too, since it may no longer match

        This is especially helpful when:
        - changing output heads
        - changing hidden sizes
        - extending pretrained encoders
        """
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def upload_predictions_to_wandb(self):
        """
        Uploads the test predictions DataFrame created in on_test_epoch_end()
        into Weights & Biases as a table.
        """
        upload_pandas_df_to_wandb(
            run_id=self.wandb_id,
            table_name="test_predictions",
            df=self.predictions_df,
        )


class NonNeuralMixin(object):
    """
    Mixin for non-neural models that still need to fit into the same
    training framework.

    Why this exists:
    - some models do not use backpropagation
    - Lightning still expects a loss tensor and optimizer interface
    - this mixin provides minimal "dummy" behavior so the rest of the
      infrastructure does not break
    """

    def training_step(self, *args, **kwargs):
        """
        Returns a tiny dummy tensor that behaves like a loss,
        purely to satisfy the training loop.
        """
        shim = torch.FloatTensor([0.0])
        shim.requires_grad = True
        return {"loss": shim}

    def configure_optimizers(self):
        """
        Dummy optimizer/scheduler for non-neural methods.

        This is mostly a compatibility shim so Lightning can run.
        """
        optimizer = torch.optim.Adam([torch.FloatTensor([])])

        def scheduler(step):
            return min(1.0, float(step + 1) / max(self.warmup_steps, 1))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler)

        return [optimizer], [
            {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        ]

    def backward(self, use_amp, loss, optimizer):
        """
        Override backward with a no-op because non-neural models
        do not backpropagate gradients.
        """
        return


class ModelTypeMixin:
    """
    Small mixin that carries flags describing what kind of model this is.

    These flags are used elsewhere in the codebase to decide:
    - which metrics to use
    - which plots to log
    - which bookkeeping behavior applies
    """

    def __init__(self):
        self.is_regressor = False
        self.is_classifier = False
        self.is_autoencoder = False
        self.is_double_encoding = False

        self.metric_class = None


class ClassificationModel(SensingModel):
    """
    Thin wrapper representing classification models.

    Main purpose:
    - pass TorchMetricClassification into SensingModel
    - mark the instance as a classifier so classifier-specific logic runs

    Examples of classifier-specific behavior:
    - cache logits/labels during train/val/test
    - compute and log ROC / PR / DET curves
    """
    def __init__(self, **kwargs) -> None:
        SensingModel.__init__(
            self,
            metric_class=TorchMetricClassification,
            **kwargs,
        )
        self.is_classifier = True


class RegressionModel(SensingModel, ModelTypeMixin):
    """
    Thin wrapper representing regression models.

    Main purpose:
    - pass TorchMetricRegression into SensingModel
    - mark the instance as a regressor

    Regression models do not use the classifier-specific curve logging.
    """
    def __init__(self, **kwargs) -> None:
        SensingModel.__init__(
            self,
            metric_class=TorchMetricRegression,
            **kwargs,
        )
        self.is_regressor = True