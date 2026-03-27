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

"""
import sys

root = "/home/hice1/ezg6/projects/Homekit2020"
if root not in sys.path:
    sys.path.insert(0, root)
    
import gc
import time

from src.models.tasks import PredictFluPos

dls = DataLoadersV1(
    taskCls=PredictFluPos,
    task_kwargs={
        "train_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/train_7_day",
        "val_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/eval_7_day",
        "test_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/test_7_day",
        "window_onset_min": 0,
        "window_onset_max": 0,
        #"shape": (10080, 8),
    },
)

task = dls.task

# Built-in task debug
#dls.debug_task_info()

# Construct loaders once for basic inspection
train_loader, val_loader, test_loader = dls.get_dataloaders()

print("\n=== DATALOADER CONSTRUCTION ===")
print("train_loader:", type(train_loader))
print("val_loader:", type(val_loader))
print("test_loader:", type(test_loader))
print("is_classification:", task.is_classification)
print("is_regression:", task.is_regression)
print("is_classification:", task.is_classification)
print("is_autoencoder:", task.is_autoencoder)
print("is_double_encoding:", task.is_double_encoding)
print("\n=== SCHEMA CHECK ===")
print("schema type:", type(task.schema))
print("schema fields:", list(task.schema.fields.keys()))


#"""
import sys

root = "/home/hice1/ezg6/projects/Homekit2020"
if root not in sys.path:
    sys.path.insert(0, root)

from src.models.tasks import get_task_with_name, PredictFluPos

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
import sys
import gc
import time
import numpy as np

root = "/home/hice1/ezg6/projects/Homekit2020"
if root not in sys.path:
    sys.path.insert(0, root)

from src.models.tasks import PredictFluPos

# NumPy compatibility for older petastorm/homekit code
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

task = PredictFluPos(
    train_path="/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/train_7_day",
    val_path="/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/eval_7_day",
    test_path="/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/test_7_day",
    window_onset_min=0,
    window_onset_max=0,
)

print("=== TASK INFO ===")
print("task type:", type(task))
print("task name:", task.get_name())
print("is_classification:", task.is_classification)
print("fields:", task.fields)
print("num fields:", len(task.fields))
print("data_shape:", task.data_shape)
print("train_path:", task.train_path)
print("val_path:", task.val_path)
print("test_path:", task.test_path)

print("\n=== SCHEMA CHECK ===")
print("schema type:", type(task.schema))
print("schema field names:", list(task.schema.fields.keys()))

for f in task.fields:
    assert f in task.schema.fields, f"Missing field: {f}"

for k, v in task.schema.fields.items():
    print(k, "dtype=", v.numpy_dtype, "shape=", v.shape)

expected = set(task.fields)
actual = set(task.schema.fields.keys())
print("missing expected fields:", expected - actual)
print("extra fields:", actual - expected)

print("\n=== LABELER CHECK ===")
print("labler type:", type(task.labler))
print("lab_results_reader type:", type(task.labler.lab_results_reader))
print("num participants with lab results:", len(task.labler.lab_results_reader.participant_ids))
print("results empty:", task.labler.lab_results_reader.results.empty)
print(task.labler.lab_results_reader.results.head())


def get_loader(task, split):
    if split == "train":
        return task.train_dataloader()
    elif split == "val":
        return task.val_dataloader()
    elif split == "test":
        return task.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")


def get_exact_samples(task, split):
    """
    Exact total samples from the task's dataset extraction path.
    """
    if split == "train":
        participant_ids, starts, x, y = task.get_train_dataset()
    elif split == "val":
        participant_ids, starts, x, y = task.get_val_dataset()
    elif split == "test":
        participant_ids, starts, x, y = task.get_test_dataset()
    else:
        raise ValueError(f"Unknown split: {split}")

    return len(y)


def inspect_first_batch(task, split):
    """
    Build a fresh loader, pull one batch, and report structure.
    """
    loader = get_loader(task, split)
    it = iter(loader)
    batch = next(it)

    print(f"\n=== FIRST {split.upper()} BATCH TEST ===")
    print("loader type:", type(loader))
    print("batch type:", type(batch))

    print(f"\n=== {split.upper()} BATCH STRUCTURE ===")
    if isinstance(batch, dict):
        print("batch keys:", list(batch.keys()))
        for k, v in batch.items():
            print(f"\nKEY: {k}")
            print("TYPE:", type(v))
            if hasattr(v, "shape"):
                print("SHAPE:", tuple(v.shape))
            elif hasattr(v, "__len__"):
                print("LEN:", len(v))
            else:
                print("VALUE:", v)

        x = batch["inputs_embeds"]
        y = batch["label"]

    elif isinstance(batch, (tuple, list)):
        print("batch length:", len(batch))
        for i, item in enumerate(batch):
            print(f"\nITEM {i}")
            print("TYPE:", type(item))
            if hasattr(item, "shape"):
                print("SHAPE:", tuple(item.shape))
            elif hasattr(item, "__len__"):
                print("LEN:", len(item))
            else:
                print("VALUE:", item)

        x, y = batch[0], batch[1]

    else:
        print(batch)
        x, y = None, None

    if x is not None and y is not None:
        print(f"\n=== FIRST {split.upper()} BATCH SUMMARY ===")
        print("x dtype:", x.dtype)
        print("y dtype:", y.dtype)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print("batch size:", y.numel())
        print("x nan count:", x.isnan().sum().item() if hasattr(x, "isnan") else "n/a")
        print("y unique:", y.unique() if hasattr(y, "unique") else "n/a")

    batch_size = y.numel() if y is not None else None

    # cleanup local refs
    del it
    del loader
    del batch
    if x is not None:
        del x
    if y is not None:
        del y
    gc.collect()
    time.sleep(1)

    return batch_size


def count_batches_and_samples(task, split):
    """
    Exact number of batches and samples by iterating once through a fresh loader.
    """
    loader = get_loader(task, split)
    it = iter(loader)

    batch_count = 0
    sample_count = 0

    try:
        while True:
            batch = next(it)
            batch_count += 1

            if isinstance(batch, dict):
                sample_count += batch["label"].numel()
            elif isinstance(batch, (tuple, list)):
                sample_count += batch[1].numel()
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")
    except StopIteration:
        pass
    finally:
        try:
            del batch
        except Exception:
            pass
        del it
        del loader
        gc.collect()
        time.sleep(1)

    return batch_count, sample_count


def estimate_positive_rate(task, split="train", num_batches=20):
    """
    Sample the first num_batches from a fresh loader and estimate class imbalance.
    """
    loader = get_loader(task, split)
    it = iter(loader)

    total = 0
    pos = 0

    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        if isinstance(batch, dict):
            y = batch["label"]
        elif isinstance(batch, (tuple, list)):
            y = batch[1]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        total += y.numel()
        pos += (y == 1).sum().item()

    rate = pos / total if total > 0 else 0.0

    # cleanup local refs
    try:
        del batch
    except Exception:
        pass
    del it
    del loader
    gc.collect()
    time.sleep(1)

    return pos, total, rate


print("\n=== DATALOADER CONSTRUCTION ===")
train_loader = task.train_dataloader()
val_loader = task.val_dataloader()
test_loader = task.test_dataloader()
print("train_loader:", type(train_loader))
print("val_loader:", type(val_loader))
print("test_loader:", type(test_loader))

# Drop these immediately; Petastorm loaders are safer when recreated fresh per pass
del train_loader, val_loader, test_loader
gc.collect()
time.sleep(1)

batch_sizes = {}
for split in ["train", "val", "test"]:
    batch_sizes[split] = inspect_first_batch(task, split)

print("\n=== EXACT TOTAL SAMPLES PER SPLIT ===")
for split in ["train", "val", "test"]:
    total_samples = get_exact_samples(task, split)
    print(split, "exact total samples:", total_samples)

print("\n=== EXACT BATCH COUNTS PER SPLIT ===")
for split in ["train", "val", "test"]:
    batch_count, sample_count = count_batches_and_samples(task, split)
    print(split, "number of batches:", batch_count)
    print(split, "samples counted through loader:", sample_count)
    print(split, "observed batch size:", batch_sizes[split])
    print("---")

print("\n=== POSITIVE RATE ESTIMATES ===")
for split in ["train", "val", "test"]:
    pos, total, rate = estimate_positive_rate(task, split=split, num_batches=20)
    print(split, "positives:", pos, "total sampled:", total, "rate:", rate)

gc.collect()
time.sleep(2)
print("finished cleanly")