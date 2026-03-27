from .core import Callback

import logging
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)



import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Dict, Any


class DistributedTrainer(Callback):
    def __init__(
        self,
        local_rank: int,
        world_size: int,
        sync_bn: bool = True,
        **kwargs
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.sync_bn = sync_bn
        self.kwargs = kwargs
        super().__init__()

    def before_fit(self):
        import time, os
        rank = int(os.environ.get("RANK", 0))
        t0 = time.time()
        print(f"[rank {rank}] DistributedTrainer.before_fit: start", flush=True)

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        print(f"[rank {rank}] DistributedTrainer.before_fit: process group ready ({time.time()-t0:.1f}s)", flush=True)

        model_to_prepare = (
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.sync_bn else self.model
        )
        print(f"[rank {rank}] DistributedTrainer.before_fit: SyncBN done ({time.time()-t0:.1f}s)", flush=True)

        self.learner.model = self.prepare_model(
            model_to_prepare,
            ddp_kwargs=self.kwargs
        )
        print(f"[rank {rank}] DistributedTrainer.before_fit: DDP model ready ({time.time()-t0:.1f}s)", flush=True)

        self.old_train_dl = self.dls.train
        self.old_valid_dl = self.dls.valid

        self.learner.dls.train = self._wrap_dl(self.dls.train)
        print(f"[rank {rank}] DistributedTrainer.before_fit: train dl wrapped ({time.time()-t0:.1f}s)", flush=True)
        self.learner.dls.valid = self._wrap_dl(self.dls.valid)
        print(f"[rank {rank}] DistributedTrainer.before_fit: valid dl wrapped ({time.time()-t0:.1f}s)", flush=True)

    def _wrap_dl(self, dl):
        return dl if isinstance(dl, DistributedDL) else self.prepare_data_loader(dl)

    def after_fit(self):
        if isinstance(self.learner.model, DistributedDataParallel):
            self.learner.model = self.learner.model.module

        self.learner.dls.train = self.old_train_dl
        self.learner.dls.valid = self.old_valid_dl

        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def prepare_model(
        self,
        model: torch.nn.Module,
        move_to_device: bool = True,
        wrap_ddp: bool = True,
        ddp_kwargs: Optional[Dict[str, Any]] = None
    ) -> torch.nn.Module:
        ddp_kwargs = ddp_kwargs or {}

        rank = self.local_rank
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        if move_to_device:
            logger.info(f"Moving model to device: {device}")
            model = model.to(device)

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

    def _is_standard_torch_dataloader(self, loader) -> bool:
        """
        True only for loaders that expose the normal torch DataLoader API
        needed by with_sampler(...).
        """
        required_attrs = [
            "sampler",
            "dataset",
            "batch_size",
            "num_workers",
            "collate_fn",
            "pin_memory",
            "drop_last",
            "timeout",
            "worker_init_fn",
        ]
        return all(hasattr(loader, attr) for attr in required_attrs)

    def prepare_data_loader(
        self,
        data_loader,
        add_dist_sampler: bool = True,
        move_to_device: bool = True
    ):
        def with_sampler(loader):
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

        # Only rebuild sampler for standard torch DataLoaders.
        # Petastorm-backed loaders do not provide the same API and should be left alone.
        if add_dist_sampler and self._is_standard_torch_dataloader(data_loader):
            data_loader = with_sampler(data_loader)
        else:
            logger.info("Skipping DistributedSampler wrapping for non-standard/Petastorm loader.")

        if move_to_device:
            if torch.cuda.is_available():
                rank = self.local_rank
                device = torch.device(f"cuda:{rank}")
            else:
                device = torch.device("cpu")

            data_loader = DistributedDL(data_loader, device)

        return data_loader


class DistributedDL(DataLoader):
    def __init__(self, base_dataloader, device: torch.device):
        self.__dict__.update(getattr(base_dataloader, "__dict__", {}))
        self.dataloader = base_dataloader
        self.device = device

    def _move_to_device(self, item):
        def try_move_device(x):
            try:
                return x.to(self.device)
            except AttributeError:
                return x

        # Petastorm/Homekit dict batch
        if isinstance(item, dict):
            return {k: try_move_device(v) for k, v in item.items()}

        # Standard tuple/list batch
        if isinstance(item, tuple):
            return tuple(try_move_device(x) for x in item)

        if isinstance(item, list):
            return [try_move_device(x) for x in item]

        # Single tensor / single object
        return try_move_device(item)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iterator = iter(self.dataloader)
        for item in iterator:
            yield self._move_to_device(item)
            
         