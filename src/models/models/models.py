"""
====================================================
Architectures For Behavioral Representation Learning     
====================================================
`Project repository available here  <https://github.com/behavioral-data/SeattleFluStudy>`_

This module contains the architectures used for behavioral representation learning in the reference paper. 
Particularly, the two main classes in the module implement a CNN architecture and the novel CNN-Transformer
architecture. 

**Classes**
    :class CNNEncoder: 
    :class CNNToTransformerEncoder:

"""

from copy import copy

from transformers import PatchTSTConfig, PatchTSTForPretraining
from typing import Dict, Tuple,  Union, Any, Optional, List, Callable
import torch
import torch.nn as nn

from sktime.classification.hybrid import HIVECOTEV2 as BaseHIVECOTEV2
import xgboost as xgb

import src.models.models.modules as modules
from src.utils import get_logger
from src.models.loops import DummyOptimizerLoop, NonNeuralLoop
from src.models.models.bases import ClassificationModel, NonNeuralMixin

from src.models.losses import build_loss_fn
from torch.utils.data.dataloader import DataLoader
from wandb.plot.roc_curve import roc_curve

logger = get_logger(__name__)


"""
 Helper functions:
"""

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


       
class CNNToTransformerClassifier(ClassificationModel):

    def __init__(self, num_attention_heads : int = 4, num_hidden_layers: int = 4,  
                kernel_sizes=[5,3,1], out_channels = [256,128,64], 
                stride_sizes=[2,2,2], dropout_rate=0.3, num_labels=2, 
                positional_encoding = False, pretrained_ckpt_path : Optional[str] = None,
                loss_fn="CrossEntropyLoss", pos_clas_weight=1, neg_class_weight=1, **kwargs) -> None:

        super().__init__(**kwargs)

        if num_hidden_layers == 0:
            self.name = "CNNClassifier"
        else:
            self.name = "CNNToTransformerClassifier"
        n_timesteps, input_features = kwargs.get("input_shape")

        self.criterion = build_loss_fn(loss_fn=loss_fn, task_type="classification")


        self.encoder = modules.CNNToTransformerEncoder(input_features, num_attention_heads, num_hidden_layers,
                                                      n_timesteps, kernel_sizes=kernel_sizes, out_channels=out_channels,
                                                      stride_sizes=stride_sizes, dropout_rate=dropout_rate, num_labels=num_labels,
                                                      positional_encoding=positional_encoding)
        
        self.head = modules.ClassificationModule(self.encoder.d_model, self.encoder.final_length, num_labels)

        if pretrained_ckpt_path:
            ckpt = torch.load(pretrained_ckpt_path)
            try:
                self.load_state_dict(ckpt['state_dict'])
            
            #TODO: Nasty hack for reverse compatability! 
            except RuntimeError:
                new_state_dict = {}
                for k,v in ckpt["state_dict"].items():
                    if not "encoder" in k :
                        new_state_dict["encoder."+k] = v
                    else:
                        new_state_dict[k] = v
                self.load_state_dict(new_state_dict, strict=False)

        self.save_hyperparameters()
        
    def forward(self, inputs_embeds,labels):
        encoding = self.encoder.encode(inputs_embeds)
        preds = self.head(encoding)
        loss =  self.criterion(preds,labels)
        return loss, preds


class MaskedCNNToTransformerClassifier(CNNToTransformerClassifier):

    def __init__(self, mask_train=True, mask_eval=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_train = mask_train
        self.mask_eval = mask_eval

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]

        if self.mask_train:
            mask = batch["mask"].bool()
            x = x[mask]
            y = y[mask]

        loss,preds = self.forward(x,y)

        self.log("train/loss", loss.item(),on_step=True)
        preds = preds.detach()

        y = y.int().detach()
        self.train_metrics.update(preds,y)

        if self.is_classifier:
            self.train_preds.append(preds.detach().cpu())
            self.train_labels.append(y.detach().cpu())

        return {"loss": loss, "preds": preds, "labels":y}

    def validation_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]


        if self.mask_eval:
            mask = batch["mask"].bool()
            x = x[mask]
            y = y[mask]

        loss,preds = self.forward(x,y)

        self.log("val/loss", loss.item(),on_step=True,sync_dist=True)


        if self.is_classifier:
            self.val_preds.append(preds.detach())
            self.val_labels.append(y.detach())

        self.val_metrics.update(preds,y)
        return {"loss":loss, "preds": preds, "labels":y}

    def test_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]
        dates = batch["end_date_str"]
        participant_ids = batch["participant_id"]

        if self.mask_eval:
            mask = batch["mask"].bool()
            x = x[mask]
            y = y[mask]

        loss,preds = self.forward(x,y)

        self.log("test/loss", loss.item(),on_step=True,sync_dist=True)


        self.test_preds.append(preds.detach())
        self.test_labels.append(y.detach())
        self.test_participant_ids.append(participant_ids)
        self.test_dates.append(dates)

        self.test_metrics.update(preds,y)
        return {"loss":loss, "preds": preds, "labels":y}



class WeakCNNToTransformerClassifier(CNNToTransformerClassifier):

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]
        y_bar = batch["weak_label"].float()

        loss,preds = self.forward(x,y_bar)

        self.log("train/loss", loss.item(),on_step=True)
        preds = preds.detach()

        y = y.int().detach()
        self.train_metrics.update(preds,y)

        if self.is_classifier:
            self.train_preds.append(preds.detach().cpu())
            self.train_labels.append(y.detach().cpu())

        return {"loss": loss, "preds": preds, "labels": y}

class TransformerClassifier(ClassificationModel):
    
    def __init__(
        self,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 4,
        dropout_rate: float = 0.,
        num_labels: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.name = "TransformerClassifier"
        n_timesteps, input_features = kwargs.get("input_shape")

        self.criterion = nn.CrossEntropyLoss()
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(input_features, num_attention_heads, dropout_rate) for _ in range(num_hidden_layers)
        ])
        
        self.head = modules.ClassificationModule(input_features, n_timesteps, num_labels)

    
    def forward(self, inputs_embeds,labels):
        x = inputs_embeds
        for l in self.blocks:
            x = l(x)

        preds = self.head(x)
        loss =  self.criterion(preds,labels)
        return loss, preds




from typing import Dict, Union, Optional, Tuple, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor


from transformers import (
    PatchTSTConfig,
    PatchTSTForPretraining,
    PatchTSTForClassification,
)

from src.models.models.bases import ClassificationModel
from src.utils import get_logger

logger = get_logger(__name__)


class PatchTSTPretrainer(pl.LightningModule):
    """
    Self-supervised PatchTST pretraining wrapper.

    Purpose
    -------
    This class is for masked patch pretraining only.
    It is intentionally separate from ClassificationModel because:
    - pretraining does not use class labels
    - pretraining loss/outputs differ from classification
    - the batch structure may be different

    Expected batch
    --------------
    At minimum:
        batch["inputs_embeds"] -> tensor of shape [B, T, C]

    Optionally:
        batch["past_observed_mask"] -> tensor of shape [B, T, C]

    Notes
    -----
    Hugging Face PatchTSTForPretraining expects `past_values` as the main input.
    This wrapper maps your repo's batch naming (`inputs_embeds`) into that API.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        learning_rate: float = 1e-3,
        warmup_steps: int = 0,
        batch_size: int = 800,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 3,
        d_model: int = 128,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        patch_length: int = 16,
        patch_stride: int = 16,
        random_mask_ratio: float = 0.5,
        channel_attention: bool = False,
        pretrained_ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        n_timesteps, input_features = input_shape
        self.name = "PatchTSTPretrainer"

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.config = PatchTSTConfig(
            num_input_channels=input_features,
            context_length=n_timesteps,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
            patch_length=patch_length,
            patch_stride=patch_stride,
            do_mask_input=True,
            mask_type="random",
            random_mask_ratio=random_mask_ratio,
            channel_attention=channel_attention,
        )

        self.model = PatchTSTForPretraining(self.config)

        if pretrained_ckpt_path:
            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretraining checkpoint. Missing: {missing}, Unexpected: {unexpected}")

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--warmup_steps", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=800)
        parser.add_argument("--num_attention_heads", type=int, default=8)
        parser.add_argument("--num_hidden_layers", type=int, default=3)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--ffn_dim", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--patch_length", type=int, default=16)
        parser.add_argument("--patch_stride", type=int, default=16)
        parser.add_argument("--random_mask_ratio", type=float, default=0.5)
        return parser

    def forward(self, inputs_embeds, past_observed_mask=None):
        """
        Returns
        -------
        loss, outputs

        `outputs` is the full Hugging Face output object, which may contain
        reconstruction-related tensors depending on model/version/config.
        """
        outputs = self.model(
            past_values=inputs_embeds,
            past_observed_mask=past_observed_mask,
        )
        return outputs.loss, outputs

    def training_step(
        self, batch, batch_idx
    ) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x = batch["inputs_embeds"].float().to(self.device)
        observed_mask = batch.get("past_observed_mask", None)
        if observed_mask is not None:
            observed_mask = observed_mask.float().to(self.device)

        loss, outputs = self.forward(x, observed_mask)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.detach().cpu())

        return {"loss": loss}

    def validation_step(
        self, batch, batch_idx
    ) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x = batch["inputs_embeds"].float().to(self.device)
        observed_mask = batch.get("past_observed_mask", None)
        if observed_mask is not None:
            observed_mask = observed_mask.float().to(self.device)

        loss, outputs = self.forward(x, observed_mask)

        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.val_losses.append(loss.detach().cpu())

        return {"loss": loss}

    def test_step(
        self, batch, batch_idx
    ) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x = batch["inputs_embeds"].float().to(self.device)
        observed_mask = batch.get("past_observed_mask", None)
        if observed_mask is not None:
            observed_mask = observed_mask.float().to(self.device)

        loss, outputs = self.forward(x, observed_mask)

        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.test_losses.append(loss.detach().cpu())

        return {"loss": loss}

    def configure_optimizers(self):
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
                "monitor": "val/loss",
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
        optimizer.step(closure=optimizer_closure)


class PatchTSTClassifier(ClassificationModel):
    """
    PatchTST classifier wrapper that fits your repo's existing classifier API.

    Purpose
    -------
    This class lets you use Hugging Face PatchTSTForClassification while
    preserving the rest of your codebase's expectations:

        forward(inputs_embeds, labels) -> (loss, preds)

    where:
    - inputs_embeds is your time-series tensor [B, T, C]
    - labels is your class target tensor
    - preds are logits

    This means it should plug into your existing ClassificationModel /
    SensingModel training, validation, and test machinery.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_attention_heads: int = 8,
        num_hidden_layers: int = 3,
        d_model: int = 128,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        patch_length: int = 16,
        patch_stride: int = 16,
        num_labels: int = 2,
        pooling_type: str = "mean",
        learning_rate: float = 1e-3,
        warmup_steps: int = 0,
        batch_size: int = 800,
        pretrained_ckpt_path: Optional[str] = None,
        classifier_ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        input_shape:
            (n_timesteps, input_features)

        pretrained_ckpt_path:
            Path to a self-supervised pretraining checkpoint. Intended for
            loading shared backbone weights into the classifier.

        classifier_ckpt_path:
            Path to a full classification checkpoint. If provided, this
            attempts to restore the full model directly.

        num_labels:
            Number of output classes. For binary classification in a
            CrossEntropy-style setup, use 2.
        """
        super().__init__(
            input_shape=input_shape,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            **kwargs,
        )

        self.name = "PatchTSTClassifier"
        n_timesteps, input_features = input_shape

        self.config = PatchTSTConfig(
            num_input_channels=input_features,
            context_length=n_timesteps,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
            patch_length=patch_length,
            patch_stride=patch_stride,
            num_targets=num_labels,
            pooling_type=pooling_type,
        )

        self.model = PatchTSTForClassification(self.config)

        # ------------------------------------------------------------------
        # Option 1: load a full classifier checkpoint directly.
        # This is for resuming a fine-tuned classifier.
        # ------------------------------------------------------------------
        if classifier_ckpt_path:
            ckpt = torch.load(classifier_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded classifier checkpoint. Missing: {missing}, Unexpected: {unexpected}")

        # ------------------------------------------------------------------
        # Option 2: initialize classifier from a pretraining checkpoint.
        # This is for transfer learning:
        # - encoder/backbone weights should be reused
        # - classification head may not match and can be skipped
        #
        # The exact module names inside Hugging Face can vary by version,
        # so strict=False is used. You may later want to inspect which
        # keys loaded cleanly in your environment.
        # ------------------------------------------------------------------
        elif pretrained_ckpt_path:
            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained checkpoint into classifier. Missing: {missing}, Unexpected: {unexpected}")

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--warmup_steps", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=800)
        parser.add_argument("--num_attention_heads", type=int, default=8)
        parser.add_argument("--num_hidden_layers", type=int, default=3)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--ffn_dim", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--patch_length", type=int, default=16)
        parser.add_argument("--patch_stride", type=int, default=16)
        parser.add_argument("--num_labels", type=int, default=2)
        parser.add_argument("--pooling_type", type=str, default="mean")
        return parser

    def forward(self, inputs_embeds, labels):
        """
        Expected input
        --------------
        inputs_embeds : Tensor
            Shape [B, T, C]

        labels : Tensor
            For CrossEntropy-style classification, shape [B]
            with integer class indices in [0, num_labels-1].

        Returns
        -------
        loss, preds
            loss  : scalar tensor
            preds : logits tensor, shape [B, num_labels]

        Why return this way?
        --------------------
        Because your base class SensingModel and all the existing classifier
        infrastructure expect:
            loss, preds = self.forward(x, y)
        """
        labels = labels.long().view(-1)

        outputs = self.model(
            past_values=inputs_embeds,
            target_values=labels,
        )

        # Hugging Face classification output stores logits in
        # `prediction_logits` for PatchTST classification.
        preds = outputs.prediction_logits
        loss = outputs.loss

        return loss, preds



    

class HIVECOTE2(NonNeuralMixin,ClassificationModel):
    
    
    def __init__(
        self,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model = BaseHIVECOTEV2(n_jobs=n_jobs)
        self.fit_loop = NonNeuralLoop()
        self.optimizer_loop = DummyOptimizerLoop()
        self.save_hyperparameters()
    
    def forward(self, inputs_embeds,labels):
        return self.base_model(inputs_embeds)

class XGBoost(xgb.XGBClassifier, NonNeuralMixin,ClassificationModel):

    def __init__(
            self,
            random_state=None,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fit_loop = NonNeuralLoop()
        self.optimizer_loop = DummyOptimizerLoop()
        self.save_hyperparameters()
        self.name = "XGBoostClassifier"
        self.random_state = random_state

    def forward(self, inputs_embeds,labels):
        raise NotImplementedError

