from .core import Callback

import logging
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)


class DistributedTrainer(Callback):
    """
    Callback that prepares the learner for Distributed Data Parallel (DDP) training.

    What this callback does:
    1. Initializes the distributed process group if needed.
    2. Moves the model to the correct local GPU for this rank.
    3. Optionally converts BatchNorm layers to SyncBatchNorm.
    4. Wraps the model in DistributedDataParallel.
    5. Replaces the train/valid dataloaders with distributed versions:
       - each rank gets a DistributedSampler
       - each batch is moved to the correct device automatically

    Assumptions:
    - The script is launched with torchrun or another DDP launcher.
    - Environment variables such as LOCAL_RANK / RANK / WORLD_SIZE are set.
    """

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        sync_bn: bool = True,
        **kwargs
    ):
        """
        Parameters
        ----------
        local_rank : int
            GPU index local to the current node, usually set by torchrun.
        world_size : int
            Total number of processes participating in DDP.
        sync_bn : bool
            If True, convert BatchNorm layers to SyncBatchNorm before wrapping in DDP.
        **kwargs
            Extra kwargs passed into DistributedDataParallel.
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.sync_bn = sync_bn
        self.kwargs = kwargs
        super().__init__()

    def before_fit(self):
        """
        Runs once before training starts.

        This is the main place where DDP setup happens.
        """
        # ----------------------------------------------------------
        # 1. Initialize the process group once per process.
        #
        # This must happen BEFORE wrapping the model in DDP.
        # torchrun usually provides all required env vars.
        # For NVIDIA GPUs, "nccl" is the standard backend.
        # ----------------------------------------------------------
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # ----------------------------------------------------------
        # 2. Prepare the model for this rank:
        #    - optionally convert BatchNorm -> SyncBatchNorm
        #    - move the model to the correct GPU
        #    - wrap in DistributedDataParallel
        # ----------------------------------------------------------
        model_to_prepare = (
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.sync_bn else self.model
        )

        self.learner.model = self.prepare_model(
            model_to_prepare,
            ddp_kwargs=self.kwargs
        )

        # ----------------------------------------------------------
        # 3. Save the original dataloaders so we can restore them
        #    after training ends.
        # ----------------------------------------------------------
        self.old_train_dl = self.dls.train
        self.old_valid_dl = self.dls.valid

        # ----------------------------------------------------------
        # 4. Replace train/valid dataloaders with distributed-aware
        #    versions:
        #    - use DistributedSampler
        #    - move batches automatically to the rank's device
        # ----------------------------------------------------------
        self.learner.dls.train = self._wrap_dl(self.dls.train)
        self.learner.dls.valid = self._wrap_dl(self.dls.valid)

    def _wrap_dl(self, dl):
        """
        Wrap a dataloader only if it is not already wrapped.
        """
        return dl if isinstance(dl, DistributedDL) else self.prepare_data_loader(dl)

    def after_fit(self):
        """
        Runs once after training ends.

        We:
        1. unwrap the DDP model back to the raw model
        2. restore the original dataloaders
        3. destroy the process group cleanly
        """
        # ----------------------------------------------------------
        # Restore the raw underlying model from DDP wrapper.
        # Only do this if we actually wrapped it.
        # ----------------------------------------------------------
        if isinstance(self.learner.model, DistributedDataParallel):
            self.learner.model = self.learner.model.module

        # ----------------------------------------------------------
        # Restore original dataloaders.
        # ----------------------------------------------------------
        self.learner.dls.train = self.old_train_dl
        self.learner.dls.valid = self.old_valid_dl

        # ----------------------------------------------------------
        # Clean up the process group.
        # ----------------------------------------------------------
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def prepare_model(
        self,
        model: torch.nn.Module,
        move_to_device: bool = True,
        wrap_ddp: bool = True,
        ddp_kwargs: Optional[Dict[str, Any]] = None
    ) -> torch.nn.Module:
        """
        Prepare the model for distributed execution.

        Steps:
        1. choose the correct device for this local rank
        2. move the model to that device
        3. wrap the model in DistributedDataParallel

        Parameters
        ----------
        model : torch.nn.Module
            Model to prepare.
        move_to_device : bool
            Whether to move the model to the correct device.
        wrap_ddp : bool
            Whether to wrap the model in DDP.
        ddp_kwargs : dict or None
            Extra kwargs for DistributedDataParallel.

        Returns
        -------
        torch.nn.Module
            Prepared model.
        """
        ddp_kwargs = ddp_kwargs or {}

        # ----------------------------------------------------------
        # local_rank determines which GPU this process should use.
        # Example:
        #   local_rank=0 -> cuda:0
        #   local_rank=1 -> cuda:1
        # ----------------------------------------------------------
        rank = self.local_rank
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # ----------------------------------------------------------
        # Set the current CUDA device for this process.
        # This is important in multi-GPU jobs.
        # ----------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        # ----------------------------------------------------------
        # Move model to the correct device.
        # ----------------------------------------------------------
        if move_to_device:
            logger.info(f"Moving model to device: {device}")
            model = model.to(device)

        # ----------------------------------------------------------
        # Wrap in DDP if there is more than one process.
        # ----------------------------------------------------------
        if wrap_ddp and self.world_size > 1:
            logger.info("Wrapping model in DistributedDataParallel.")
            if torch.cuda.is_available():
                model = DistributedDataParallel(
                    model,
                    device_ids=[rank],
                    output_device=rank,
                    **ddp_kwargs
                )
            else:
                model = DistributedDataParallel(model, **ddp_kwargs)

        return model

    def prepare_data_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        add_dist_sampler: bool = True,
        move_to_device: bool = True
    ) -> torch.utils.data.DataLoader:
        """
        Prepare a DataLoader for distributed execution.

        Main ideas:
        - Replace its sampler with DistributedSampler so each rank sees
          a different shard of the dataset.
        - Optionally move each batch to the correct device automatically.

        Parameters
        ----------
        data_loader : DataLoader
            Original dataloader.
        add_dist_sampler : bool
            Whether to add a DistributedSampler.
        move_to_device : bool
            Whether to wrap the dataloader so its items are moved to device.

        Returns
        -------
        DataLoader
            A distributed-aware dataloader.
        """

        def with_sampler(loader):
            """
            Build a new DataLoader using the same settings but with
            DistributedSampler instead of the original sampler.
            """
            # ------------------------------------------------------
            # If the original loader uses SequentialSampler, then it
            # was not shuffling. Otherwise, assume shuffle=True.
            #
            # In DDP, shuffling is controlled by DistributedSampler,
            # so DataLoader(shuffle=...) itself must be False.
            # ------------------------------------------------------
            shuffle = not isinstance(loader.sampler, SequentialSampler)

            data_loader_args = {
                "dataset": loader.dataset,
                "batch_size": loader.batch_size,
                "shuffle": False,
                "num_workers": loader.num_workers,
                "collate_fn": loader.collate_fn,
                "pin_memory": loader.pin_memory,
                "drop_last": loader.drop_last,
                "timeout": loader.timeout,
                "worker_init_fn": loader.worker_init_fn,
                "sampler": DistributedSampler(loader.dataset, shuffle=shuffle),
            }

            return DataLoader(**data_loader_args)

        # ----------------------------------------------------------
        # Add DistributedSampler if requested.
        # ----------------------------------------------------------
        if add_dist_sampler:
            data_loader = with_sampler(data_loader)

        # ----------------------------------------------------------
        # Wrap dataloader so each yielded batch is moved to this rank's
        # device automatically.
        # ----------------------------------------------------------
        if move_to_device:
            if torch.cuda.is_available():
                rank = self.local_rank
                device = torch.device(f"cuda:{rank}")
            else:
                device = torch.device("cpu")

            data_loader = DistributedDL(data_loader, device)

        return data_loader


class DistributedDL(DataLoader):
    """
    Lightweight wrapper around an existing DataLoader.

    Its purpose is simple:
    - iterate through the original dataloader
    - move every item in every batch to the target device
    """

    def __init__(self, base_dataloader: DataLoader, device: torch.device):
        # Reuse most attributes from the original dataloader.
        self.__dict__.update(getattr(base_dataloader, "__dict__", {}))
        self.dataloader = base_dataloader
        self.device = device

    def _move_to_device(self, item):
        """
        Move each element in the batch tuple to the target device if possible.
        """
        def try_move_device(x):
            try:
                x = x.to(self.device)
            except AttributeError:
                logger.debug(
                    f"Item {x} cannot be moved to device {self.device}."
                )
            return x

        return tuple(try_move_device(x) for x in item)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iterator = iter(self.dataloader)

        for item in iterator:
            yield self._move_to_device(item)