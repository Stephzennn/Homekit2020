import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DataLoadersV1:
    def __init__(
        self,
        datasetCls=None,
        dataset_kwargs: dict = None,
        taskCls=None,
        task_kwargs: dict = None,
        batch_size: int = 32,
        workers: int = 2,
        collate_fn=None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
    ):
        super().__init__()

        self.datasetCls = datasetCls
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.taskCls = taskCls
        self.task_kwargs = dict(task_kwargs or {})
        self.batch_size = batch_size
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        

        if self.datasetCls is None and self.taskCls is None:
            raise ValueError("Provide either datasetCls or taskCls")

        if self.datasetCls is not None and self.taskCls is not None:
            raise ValueError("Provide only one of datasetCls or taskCls")

        # Dataset mode: wrapper controls split
        self.dataset_kwargs.pop("split", None)

        # Task mode setup
        self.task = None
        if self.taskCls is not None:
            if not hasattr(np, "unicode_"):
                np.unicode_ = np.str_
            self.task = self.taskCls(**self.task_kwargs)
        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()
    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._make_dloader("val", shuffle=self.shuffle_val)

    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)
    """
    def get_dataloaders(self):
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        test_loader = self.test_dataloader()
        return train_loader, val_loader, test_loader
    """
    def get_dataloaders(self):
        return self.train, self.valid, self.test

    def _make_dloader(self, split, shuffle=False):
        # Task-backed mode: use Homekit/Petastorm task directly
        if self.task is not None:
            if split == "train":
                return self.task.train_dataloader()
            elif split == "val":
                return self.task.val_dataloader()
            elif split == "test":
                return self.task.test_dataloader()
            else:
                raise ValueError(f"Unknown split: {split}")

        # Standard PyTorch Dataset mode
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)

        if len(dataset) == 0:
            return None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def debug_task_info(self, print_schema=True, print_labeler=True):
        if self.task is None:
            raise ValueError("debug_task_info is only available in task mode")

        print(type(self.task))
        print(self.task.get_name())
        print("is_classification:", self.task.is_classification)
       
        print("fields:", self.task.fields)
        print("num fields:", len(self.task.fields))
        print("data_shape:", self.task.data_shape)
        print("train_path:", self.task.train_path)
        print("val_path:", self.task.val_path)
        print("test_path:", self.task.test_path)

        if print_schema:
            print(type(self.task.schema))
            print("schema field names:")
            print(list(self.task.schema.fields.keys()))

            print("\nfield -> numpy dtype")
            for k, v in self.task.schema.fields.items():
                print(k, v.numpy_dtype, v.shape)

            expected = set(self.task.fields)
            actual = set(self.task.schema.fields.keys())

            print("missing expected fields:", expected - actual)
            print("extra fields:", actual - expected)

            print("\n=== SCHEMA CHECK ===")
            print("schema type:", type(self.task.schema))
            print("schema fields:", list(self.task.schema.fields.keys()))

            for f in self.task.fields:
                assert f in self.task.schema.fields, f"Missing field: {f}"

            for k, v in self.task.schema.fields.items():
                print(k, "dtype=", v.numpy_dtype, "shape=", v.shape)

        if print_labeler:
            print("\n=== LABELER CHECK ===")
            print("labler type:", type(self.task.labler))
            print("lab_results_reader type:", type(self.task.labler.lab_results_reader))
            print("num participants with lab results:", len(self.task.labler.lab_results_reader.participant_ids))
            print("results empty:", self.task.labler.lab_results_reader.results.empty)
            print(self.task.labler.lab_results_reader.results.head())
    def add_dl(self, test_data, batch_size=None, **kwargs):
        if isinstance(test_data, DataLoader):
            return test_data

        if batch_size is None:
            batch_size = self.batch_size

        if self.task is not None:
            raise NotImplementedError("add_dl is not implemented in task mode")

        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)

        # preserve original intent as closely as possible
        if hasattr(self.train, "new"):
            return self.train.new(test_data, batch_size, **kwargs)

        return DataLoader(
            test_data,
            batch_size=batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            **kwargs
        )
    def _unpack_batch(self, batch):
        # New task-backed Homekit format
        if isinstance(batch, dict):
            xb = batch["inputs_embeds"]
            yb = batch["label"]
            return xb, yb

        # Old tuple/list format
        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError(f"Batch has insufficient length: {len(batch)}")
            return batch[0], batch[1]

        raise TypeError(f"Unsupported batch type: {type(batch)}")
    
    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DictDataset(Dataset):
    """
    Simple in-memory torch dataset for batches shaped like dicts.

    Each field must have the same first dimension length.
    Example item:
        {
            "inputs_embeds": tensor [T, C],
            "label": tensor [],
            "participant_id": ...,
            "end_date_str": ...
        }
    """
    def __init__(self, data_dict):
        self.data = data_dict
        self.length = len(next(iter(data_dict.values())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {}
        for k, v in self.data.items():
            item[k] = v[idx]
        return item


class TupleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class DataLoadersV2:
    """
    Version that can operate in two modes:

    1. datasetCls mode:
       behaves like your old CSV-style DataLoaders and builds normal torch DataLoaders
       from a torch Dataset class.

    2. taskCls mode:
       expects a Petastorm/parquet-backed task, reads the full split into memory once,
       converts it into a normal torch Dataset, then returns a normal torch DataLoader.

    This means:
    - dls.train / dls.valid / dls.test are normal torch DataLoaders
    - len(dls.train) works directly
    - no LoaderWithLen wrapper is needed
    """

    def __init__(
        self,
        datasetCls=None,
        dataset_kwargs: dict = None,
        taskCls=None,
        task_kwargs: dict = None,
        batch_size: int = 128,
        workers: int = 0,
        collate_fn=None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
    ):
        super().__init__()

        self.datasetCls = datasetCls
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.taskCls = taskCls
        self.task_kwargs = dict(task_kwargs or {})
        self.batch_size = batch_size
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        if self.datasetCls is None and self.taskCls is None:
            raise ValueError("Provide either datasetCls or taskCls")
        if self.datasetCls is not None and self.taskCls is not None:
            raise ValueError("Provide only one of datasetCls or taskCls")

        self.train = self._make_dloader("train", shuffle=self.shuffle_train)
        self.valid = self._make_dloader("val", shuffle=self.shuffle_val)
        self.test = self._make_dloader("test", shuffle=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test

    def get_dataloaders(self):
        return self.train, self.valid, self.test

    def add_dl(self, test_data, batch_size=None, **kwargs):
        if isinstance(test_data, DataLoader):
            return test_data

        if batch_size is None:
            batch_size = self.batch_size

        if not isinstance(test_data, Dataset):
            raise TypeError("test_data must be a torch Dataset or DataLoader")

        return DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal loader builder
    # ------------------------------------------------------------------
    def _make_dloader(self, split, shuffle=False):
        # --------------------------------------------------------------
        # Mode 1: regular torch Dataset class (your old CSV-style path)
        # --------------------------------------------------------------
        if self.datasetCls is not None:
            dataset = self.datasetCls(**self.dataset_kwargs, split=split)
            if len(dataset) == 0:
                return None

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.workers,
                collate_fn=self.collate_fn,
            )

        # --------------------------------------------------------------
        # Mode 2: taskCls path (Petastorm/parquet)
        # Read the whole split once, convert to a normal torch dataset,
        # then return a standard torch DataLoader.
        # --------------------------------------------------------------
        dataset = self._materialize_task_split(split)
        if len(dataset) == 0:
            return None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    # ------------------------------------------------------------------
    # Task/parquet materialization
    # ------------------------------------------------------------------
    def _materialize_task_split(self, split):
        if not hasattr(np, "unicode_"):
            np.unicode_ = np.str_

        task = self.taskCls(**self.task_kwargs)

        if hasattr(task, "batch_size"):
            task.batch_size = 1

        if split == "train":
            loader = task.train_dataloader()
        elif split == "val":
            loader = task.val_dataloader()
        elif split == "test":
            loader = task.test_dataloader()
        else:
            raise ValueError(f"Unknown split: {split}")

        if loader is None:
            return TupleDataset(torch.empty(0), torch.empty(0))

        xs = []
        ys = []

        for batch in loader:
            if not isinstance(batch, dict):
                raise TypeError(f"Expected dict batch from task loader, got {type(batch)}")

            x = batch["inputs_embeds"]
            y = batch["label"]

            # If batch dimension exists, split into individual samples
            if torch.is_tensor(x):
                if x.ndim >= 1:
                    for i in range(len(x)):
                        xs.append(x[i].clone())
                else:
                    xs.append(x.clone())
            else:
                x = torch.as_tensor(x)
                for i in range(len(x)):
                    xs.append(x[i].clone())

            if torch.is_tensor(y):
                if y.ndim >= 1:
                    for i in range(len(y)):
                        ys.append(y[i].clone())
                else:
                    ys.append(y.clone())
            else:
                y = torch.as_tensor(y)
                for i in range(len(y)):
                    ys.append(y[i].clone())

        if len(xs) == 0:
            return TupleDataset(torch.empty(0), torch.empty(0))

        x_tensor = torch.stack(xs, dim=0)
        y_tensor = torch.stack(ys, dim=0)

        return TupleDataset(x_tensor, y_tensor)




import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TensorTupleDataset(Dataset):
    """
    Minimal torch Dataset that returns (x, y), matching the original
    DataLoaders batch contract expected downstream.

    x: torch.Tensor of shape [N, ...]
    y: torch.Tensor of shape [N, ...]
    """

    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        if len(x_tensor) != len(y_tensor):
            raise ValueError(
                f"x and y must have the same number of samples, got "
                f"{len(x_tensor)} and {len(y_tensor)}"
            )
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def new(self, raw_data):
        """
        Keeps compatibility with original add_dl behavior.

        raw_data is expected to be:
            - another Dataset, or
            - tuple/list of (x, y)
        """
        if isinstance(raw_data, Dataset):
            return raw_data

        if isinstance(raw_data, (tuple, list)) and len(raw_data) == 2:
            x, y = raw_data

            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)

            return TensorTupleDataset(x, y)

        raise TypeError(
            "raw_data must be a Dataset or a tuple/list of (x, y)"
        )


class _LoaderWithNew:
    """
    Small wrapper so dls.train.new(...) works like the original style.

    This helps preserve add_dl compatibility if downstream code expects
    self.train.new(dataset, batch_size=..., ...) to exist.
    """

    def __init__(self, loader, collate_fn=None, workers=0):
        self._loader = loader
        self.collate_fn = collate_fn
        self.workers = workers

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset

    def new(self, dataset, batch_size, **kwargs):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._loader, name)


class DataLoadersV3:
    """
    Parquet/task-based counterpart to the original DataLoaders.

    Design goal:
    - Interchangeable with the original DataLoaders downstream
    - Only difference is the source of data
    - Internally consumes task/parquet/Petastorm
    - Externally exposes normal torch DataLoaders returning (xb, yb)

    Expected task behavior:
    - taskCls(**task_kwargs)
    - task.train_dataloader()
    - task.val_dataloader()
    - task.test_dataloader()

    Expected task batch format:
    - dict with keys:
        "inputs_embeds"
        "label"

    Those are converted into a normal tuple dataset:
        (x, y)
    """

    def __init__(
        self,
        taskCls,
        task_kwargs: dict = None,
        batch_size: int = 128,
        workers: int = 0,
        collate_fn=None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
    ):
        super().__init__()

        if taskCls is None:
            raise ValueError("DataLoadersV3 requires taskCls (parquet/task source).")

        self.taskCls = taskCls
        self.task_kwargs = dict(task_kwargs or {})

        # Keep these names aligned with the original DataLoaders
        self.datasetCls = None
        self.dataset_kwargs = None
        self.batch_size = batch_size
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        # Build normal torch loaders
        train_loader = self._make_dloader(split="train", shuffle=self.shuffle_train)
        valid_loader = self._make_dloader(split="val", shuffle=self.shuffle_val)
        test_loader = self._make_dloader(split="test", shuffle=False)

        # Wrap so .dataset and .new are available in the same spirit as original
        self.train = None if train_loader is None else _LoaderWithNew(
            train_loader, collate_fn=self.collate_fn, workers=self.workers
        )
        self.valid = None if valid_loader is None else _LoaderWithNew(
            valid_loader, collate_fn=self.collate_fn, workers=self.workers
        )
        self.test = None if test_loader is None else _LoaderWithNew(
            test_loader, collate_fn=self.collate_fn, workers=self.workers
        )

    # ------------------------------------------------------------------
    # CLI compatibility
    # ------------------------------------------------------------------
    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    # ------------------------------------------------------------------
    # Public API: same style as original
    # ------------------------------------------------------------------
    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test

    def get_dataloaders(self):
        return self.train, self.valid, self.test

    def add_dl(self, test_data, batch_size=None, **kwargs):
        """
        Mimic original add_dl behavior as closely as possible.

        Accepts:
        - DataLoader -> returns it unchanged
        - Dataset    -> wraps it in a torch DataLoader
        - (x, y)     -> if self.train.dataset.new(...) exists, use it

        This is important so downstream code can treat original DataLoaders
        and DataLoadersV3 similarly.
        """
        if isinstance(test_data, DataLoader):
            return test_data

        if batch_size is None:
            batch_size = self.batch_size

        # If already a Dataset, use directly
        if isinstance(test_data, Dataset):
            dataset = test_data

        else:
            # Try original-style conversion path
            if self.train is not None and hasattr(self.train.dataset, "new"):
                dataset = self.train.dataset.new(test_data)
            else:
                raise TypeError(
                    "test_data must be a DataLoader, a Dataset, or something "
                    "convertible through train.dataset.new(...)"
                )

        # Follow original spirit: use self.train.new(...) if available
        if self.train is not None and hasattr(self.train, "new"):
            return self.train.new(dataset, batch_size=batch_size, **kwargs)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_dloader(self, split, shuffle=False):
        dataset = self._materialize_task_split(split=split)
        if dataset is None or len(dataset) == 0:
            return None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def _materialize_task_split(self, split):
        """
        Instantiate the task, read the requested split fully, and convert
        it into an in-memory torch Dataset returning (x, y).

        This is what makes the output interchangeable with the original
        CSV-based DataLoaders.
        """
        if not hasattr(np, "unicode_"):
            np.unicode_ = np.str_

        task = self.taskCls(**self.task_kwargs)

        # Force underlying task loader to emit simple small batches if possible.
        # This makes flattening into samples easier and more predictable.
        if hasattr(task, "batch_size"):
            task.batch_size = 1

        if split == "train":
            loader = task.train_dataloader()
        elif split == "val":
            loader = task.val_dataloader()
        elif split == "test":
            loader = task.test_dataloader()
        else:
            raise ValueError(f"Unknown split: {split}")

        if loader is None:
            return None

        xs = []
        ys = []

        for batch in loader:
            if not isinstance(batch, dict):
                raise TypeError(
                    f"Expected task loader to yield dict batches, got {type(batch)}"
                )

            if "inputs_embeds" not in batch or "label" not in batch:
                raise KeyError(
                    "Task batch must contain 'inputs_embeds' and 'label' keys"
                )

            x = batch["inputs_embeds"]
            y = batch["label"]

            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)

            # Usually batch_size=1 because we force task.batch_size = 1
            # but handle general batch sizes too.
            for i in range(len(x)):
                xs.append(x[i].clone())
                ys.append(y[i].clone())

        if len(xs) == 0:
            return None

        x_tensor = torch.stack(xs, dim=0)
        y_tensor = torch.stack(ys, dim=0)

        return TensorTupleDataset(x_tensor, y_tensor)


import sys

root = "/home/hice1/ezg6/projects/Homekit2020"
if root not in sys.path:
    sys.path.insert(0, root)
    
import gc
import time

from src.models.tasks import PredictFluPos


#"train_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/train_7_day",
#"val_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/eval_7_day",
#"test_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/test_7_day",

dls = DataLoadersV3(
    taskCls=PredictFluPos,
    task_kwargs={
        "train_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/train_7_day",
        "val_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/eval_7_day",
        "test_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/test_7_day",
        "fields": [
            "heart_rate",
            "missing_heart_rate",
            "missing_steps",
            "sleep_classic_0",
            "sleep_classic_1",
            "sleep_classic_2",
            "sleep_classic_3",
            "steps",
        ],
        "batch_size": 1,   # internal task loader batch; outer torch loader rebatches
        "activity_level": "minute",
    },
    batch_size=32,         # outer normal torch DataLoader batch size
    workers=0,
)

task = dls

# Construct loaders once for basic inspection
train_loader, val_loader, test_loader = dls.get_dataloaders()

print("\n=== DATALOADER CONSTRUCTION ===")
print("dls type:", type(dls))
print("train_loader:", type(train_loader))
print("val_loader:", type(val_loader))
print("test_loader:", type(test_loader))

print("train_loader len:", len(train_loader) if train_loader is not None else None)
print("val_loader len:", len(val_loader) if val_loader is not None else None)
print("test_loader len:", len(test_loader) if test_loader is not None else None)

print("\n=== DATALOADERSV3 ATTRIBUTES ===")
attrs_to_check = [
    "taskCls",
    "task_kwargs",
    "datasetCls",
    "dataset_kwargs",
    "batch_size",
    "workers",
    "collate_fn",
    "shuffle_train",
    "shuffle_val",
    "train",
    "valid",
    "test",
]
for attr in attrs_to_check:
    if hasattr(dls, attr):
        val = getattr(dls, attr)
        print(f"{attr}: type={type(val)}")
    else:
        print(f"{attr}: MISSING")

print("\n=== METHOD CHECK ===")
methods_to_check = [
    "train_dataloader",
    "val_dataloader",
    "test_dataloader",
    "get_dataloaders",
    "add_dl",
    "add_cli",
]
for m in methods_to_check:
    print(f"{m}: {hasattr(dls, m)}")

print("\n=== OPTIONAL METADATA CHECK ===")
for attr in ["vars", "len", "c"]:
    if hasattr(dls, attr):
        print(f"{attr}: {getattr(dls, attr)}")
    else:
        print(f"{attr}: not attached")

def inspect_one_batch(loader, name="loader"):
    print(f"\n=== INSPECTING {name.upper()} ===")

    if loader is None:
        print("Loader is None")
        return

    batch = next(iter(loader))
    print("batch type:", type(batch))

    if isinstance(batch, dict):
        print("batch keys:", list(batch.keys()))
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print(f"{k}: type={type(v)}, shape={tuple(v.shape)}, dtype={getattr(v, 'dtype', None)}")
            elif isinstance(v, (list, tuple)):
                print(f"{k}: type={type(v)}, len={len(v)}")
                if len(v) > 0:
                    print(f"  first element type: {type(v[0])}")
            else:
                print(f"{k}: type={type(v)}, value_sample={v}")

    elif isinstance(batch, (list, tuple)):
        print("batch len:", len(batch))
        for i, v in enumerate(batch):
            if hasattr(v, "shape"):
                print(f"[{i}]: type={type(v)}, shape={tuple(v.shape)}, dtype={getattr(v, 'dtype', None)}")
            elif isinstance(v, (list, tuple)):
                print(f"[{i}]: type={type(v)}, len={len(v)}")
                if len(v) > 0:
                    print(f"    first element type: {type(v[0])}")
            else:
                print(f"[{i}]: type={type(v)}, value_sample={v}")
    else:
        print("unknown batch format:", batch)

    return batch

train_batch = inspect_one_batch(train_loader, "train_loader")
val_batch = inspect_one_batch(val_loader, "val_loader")
test_batch = inspect_one_batch(test_loader, "test_loader")

print("\n=== TUPLE COMPATIBILITY CHECK ===")
if train_loader is not None:
    xb, yb = next(iter(train_loader))
    print("xb type:", type(xb), "shape:", getattr(xb, "shape", None), "dtype:", getattr(xb, "dtype", None))
    print("yb type:", type(yb), "shape:", getattr(yb, "shape", None), "dtype:", getattr(yb, "dtype", None))

print("\n=== DIRECT METHOD RETURN CHECK ===")
train_loader_2 = dls.train_dataloader()
val_loader_2 = dls.val_dataloader()
test_loader_2 = dls.test_dataloader()

print("train_dataloader() type:", type(train_loader_2))
print("val_dataloader() type:", type(val_loader_2))
print("test_dataloader() type:", type(test_loader_2))

print("len(train_dataloader()):", len(train_loader_2) if train_loader_2 is not None else None)
print("len(val_dataloader()):", len(val_loader_2) if val_loader_2 is not None else None)
print("len(test_dataloader()):", len(test_loader_2) if test_loader_2 is not None else None)

print("\n=== add_dl CHECK WITH EXISTING DATASET ===")
if train_loader is not None:
    train_dataset = train_loader.dataset
    extra_loader = dls.add_dl(train_dataset, batch_size=4)
    print("extra_loader type:", type(extra_loader))
    print("extra_loader len:", len(extra_loader))
    extra_batch = next(iter(extra_loader))
    print("extra_batch type:", type(extra_batch))
    if isinstance(extra_batch, (tuple, list)):
        print("extra_batch len:", len(extra_batch))
        for i, v in enumerate(extra_batch):
            print(f"  [{i}] shape:", getattr(v, "shape", None), "dtype:", getattr(v, "dtype", None))

print("\n=== add_dl CHECK WITH RAW (x, y) ===")
if train_loader is not None:
    xb, yb = next(iter(train_loader))
    raw_loader = dls.add_dl((xb, yb), batch_size=2)
    print("raw_loader type:", type(raw_loader))
    print("raw_loader len:", len(raw_loader))
    raw_batch = next(iter(raw_loader))
    print("raw_batch type:", type(raw_batch))
    if isinstance(raw_batch, (tuple, list)):
        print("raw_batch len:", len(raw_batch))
        for i, v in enumerate(raw_batch):
            print(f"  [{i}] shape:", getattr(v, "shape", None), "dtype:", getattr(v, "dtype", None))