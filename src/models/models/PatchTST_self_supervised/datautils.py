


import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

# ------------------------------------------------------------------
# Import the custom dataloader wrapper used by this PatchTST project.
# This helper likely builds train / valid / test DataLoader objects
# around a dataset class.
# ------------------------------------------------------------------
from src.data.datamodule import DataLoaders
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader, Dataset
import numpy as np


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
            "label": tensor [] (scalar) or tensor [T, C] when autoencoder_label=True,
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
        autoencoder_label: bool = False,
        splits=('train', 'val', 'test'),
        label_filter: str = 'all',
        neg_subsample_ratio: int = 0,
        seed: int = 42,
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
        self.autoencoder_label = autoencoder_label

        if label_filter not in ('all', 'positive', 'negative'):
            raise ValueError(f"label_filter must be 'all', 'positive', or 'negative'; got '{label_filter}'")
        self.label_filter = label_filter
        self.neg_subsample_ratio = neg_subsample_ratio
        self.seed = seed

        if self.datasetCls is None and self.taskCls is None:
            raise ValueError("Provide either datasetCls or taskCls")
        if self.datasetCls is not None and self.taskCls is not None:
            raise ValueError("Provide only one of datasetCls or taskCls")

        self.train = self._make_dloader("train", shuffle=self.shuffle_train) if 'train' in splits else None
        self.valid = self._make_dloader("val", shuffle=self.shuffle_val) if 'val' in splits else None
        self.test = self._make_dloader("test", shuffle=False) if 'test' in splits else None

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

        # Apply label-based filtering to the train split only.
        # 'positive' keeps samples with label >= 1; 'negative' keeps label == 0.
        if split == 'train' and self.label_filter != 'all':
            dataset = self._filter_by_label(dataset, self.label_filter)
            if len(dataset) == 0:
                import warnings
                warnings.warn(
                    f"label_filter='{self.label_filter}' removed all training samples. "
                    "Check that the dataset contains samples with the requested label."
                )
                return None

        # Negative undersampling: keep at most neg_subsample_ratio negatives per positive.
        if split == 'train' and self.neg_subsample_ratio > 0:
            dataset = self._subsample_negatives(dataset, self.neg_subsample_ratio, self.seed)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    # ------------------------------------------------------------------
    # Label filtering
    # ------------------------------------------------------------------
    def _filter_by_label(self, dataset: 'DictDataset', mode: str) -> 'DictDataset':
        """
        Return a new DictDataset containing only the samples whose label
        matches `mode`.

        Parameters
        ----------
        dataset : DictDataset
            Fully materialized dataset with a 'label' key.
        mode : str
            'positive' — keep samples where label >= 1 (positive class).
            'negative' — keep samples where label == 0 (healthy/negative class).

        Returns
        -------
        DictDataset
            Filtered dataset.
        """
        if 'label' not in dataset.data:
            raise KeyError(
                "label_filter requires a 'label' key in the dataset, but none was found."
            )

        labels = dataset.data['label']

        # Support both scalar labels (shape [N]) and per-timestep labels.
        # For scalar tensors we compare directly; for multi-dim we use the
        # first element (index 0) per sample as the "window label".
        if torch.is_tensor(labels):
            flat_labels = labels.view(len(labels), -1)[:, 0]  # shape [N]
            if mode == 'positive':
                mask = flat_labels >= 1
            else:  # 'negative'
                mask = flat_labels == 0
            indices = mask.nonzero(as_tuple=False).squeeze(1).tolist()
        else:
            # List of scalars / objects
            if mode == 'positive':
                indices = [i for i, v in enumerate(labels) if float(v) >= 1]
            else:
                indices = [i for i, v in enumerate(labels) if float(v) == 0]

        import os
        rank = int(os.environ.get("RANK", 0))
        print(
            f"[rank {rank}] label_filter='{mode}': "
            f"{len(indices)} / {len(dataset)} training samples retained.",
            flush=True,
        )

        filtered = {}
        for k, v in dataset.data.items():
            if torch.is_tensor(v):
                filtered[k] = v[indices]
            elif isinstance(v, np.ndarray):
                filtered[k] = v[indices]
            else:
                filtered[k] = [v[i] for i in indices]

        return DictDataset(filtered)

    def _subsample_negatives(self, dataset: 'DictDataset', ratio: int, seed: int) -> 'DictDataset':
        """Keep at most `ratio` negatives per positive in the training set."""
        labels = dataset.data['label']
        if torch.is_tensor(labels):
            flat_labels = labels.view(len(labels), -1)[:, 0]
            pos_idx = (flat_labels >= 1).nonzero(as_tuple=False).squeeze(1).tolist()
            neg_idx = (flat_labels == 0).nonzero(as_tuple=False).squeeze(1).tolist()
        else:
            pos_idx = [i for i, v in enumerate(labels) if float(v) >= 1]
            neg_idx = [i for i, v in enumerate(labels) if float(v) == 0]

        keep_n = min(len(neg_idx), len(pos_idx) * ratio)
        rng = np.random.default_rng(seed)
        kept_neg = rng.choice(neg_idx, size=keep_n, replace=False).tolist()

        indices = sorted(pos_idx + kept_neg)

        import os
        rank = int(os.environ.get("RANK", 0))
        print(
            f"[rank {rank}] neg_subsample_ratio={ratio}: "
            f"kept {len(pos_idx)} pos + {keep_n} neg "
            f"(from {len(neg_idx)}) = {len(indices)} total training samples.",
            flush=True,
        )

        subsampled = {}
        for k, v in dataset.data.items():
            if torch.is_tensor(v):
                subsampled[k] = v[indices]
            elif isinstance(v, np.ndarray):
                subsampled[k] = v[indices]
            else:
                subsampled[k] = [v[i] for i in indices]

        return DictDataset(subsampled)

    # ------------------------------------------------------------------
    # Task/parquet materialization
    # ------------------------------------------------------------------
    def _materialize_task_split(self, split):
        """
        Instantiate the task, get the Petastorm-backed loader for the split,
        read all samples into memory, and wrap them as a normal torch Dataset.
        """
        import time
        import os
        rank = int(os.environ.get("RANK", 0))

        if not hasattr(np, "unicode_"):
            np.unicode_ = np.str_

        t0 = time.time()
        #print(f"[rank {rank}] _materialize_task_split({split}): instantiating task...")
        task = self.taskCls(**self.task_kwargs)
        #print(f"[rank {rank}] _materialize_task_split({split}): task ready ({time.time()-t0:.1f}s)")

        if hasattr(task, "batch_size"):
            task.batch_size = self.batch_size
            #print(f"[rank {rank}] _materialize_task_split({split}): using batch_size={self.batch_size}")

        t1 = time.time()
        if split == "train":
            loader = task.train_dataloader()
        elif split == "val":
            loader = task.val_dataloader()
        elif split == "test":
            loader = task.test_dataloader()
        else:
            raise ValueError(f"Unknown split: {split}")
        #print(f"[rank {rank}] _materialize_task_split({split}): loader ready ({time.time()-t1:.1f}s)")

        if loader is None:
            return DictDataset({})

        samples = []
        t2 = time.time()
        #print(f"[rank {rank}] _materialize_task_split({split}): reading batches from loader...")

        for batch_idx, batch in enumerate(loader):
            if batch_idx % 20 == 0:
                elapsed = time.time() - t2
                #print(f"[rank {rank}]   {split} batch {batch_idx}, samples so far: {len(samples)}, elapsed: {elapsed:.1f}s")

            # batch is usually a dict from Petastorm/PyTorch integration
            if isinstance(batch, dict):
                batch_size = None
                for v in batch.values():
                    try:
                        batch_size = len(v)
                        break
                    except TypeError:
                        continue

                if batch_size is None:
                    samples.append(batch)
                else:
                    for i in range(batch_size):
                        sample = {}
                        for k, v in batch.items():
                            try:
                                sample[k] = v[i]
                            except Exception:
                                sample[k] = v
                        samples.append(sample)

            else:
                raise TypeError(
                    f"Expected task loader to yield dict batches, got {type(batch)}"
                )

        #print(f"[rank {rank}] _materialize_task_split({split}): done reading — {len(samples)} samples in {time.time()-t2:.1f}s")

        if len(samples) == 0:
            return DictDataset({})

        # Convert list[dict] -> dict[list] -> dict[tensor-or-list]
        t3 = time.time()
        #print(f"[rank {rank}] _materialize_task_split({split}): stacking tensors...")
        keys = samples[0].keys()
        columns = {k: [sample[k] for sample in samples] for k in keys}

        materialized = {}
        for k, vals in columns.items():
            first_val = vals[0]

            # tensor case
            if torch.is_tensor(first_val):
                materialized[k] = torch.stack(vals, dim=0)

            # numpy array case
            elif isinstance(first_val, np.ndarray):
                materialized[k] = torch.from_numpy(np.stack(vals, axis=0))

            # numeric scalar case
            elif isinstance(first_val, (int, float, np.integer, np.floating)):
                materialized[k] = torch.tensor(vals)

            # strings / ids / dates / objects: keep as python list
            else:
                materialized[k] = vals

        #for k, v in materialized.items():
        #    shape = v.shape if hasattr(v, "shape") else f"list[{len(v)}]"
        #    print(f"[rank {rank}]   materialized['{k}']: {shape}")
        #print(f"[rank {rank}] _materialize_task_split({split}): stacking done ({time.time()-t3:.1f}s)")

        # Autoencoder mode: replace the scalar classification label with a
        # zero tensor matching inputs_embeds shape so yb is never a scalar.
        # We use zeros (not a clone) to avoid doubling memory and wasting
        # GPU bandwidth — PatchMaskCB overwrites yb with xb_patch anyway.
        if self.autoencoder_label and "inputs_embeds" in materialized:
            materialized["label"] = torch.zeros_like(materialized["inputs_embeds"])

        return DictDataset(materialized)

# ------------------------------------------------------------------
# Import dataset classes used for different forecasting benchmarks:
# - Dataset_ETT_minute
# - Dataset_ETT_hour
# - Dataset_Custom
# and any other helpers defined in pred_dataset.py
# ------------------------------------------------------------------
from src.data.pred_dataset import *


# ------------------------------------------------------------------
# List of supported dataset names.
# The code will only accept one of these strings as params.dset.
# ------------------------------------------------------------------
DSETS = [
    'ettm1', 'ettm2', 'etth1', 'etth2',
    'electricity', 'traffic', 'illness',
    'weather', 'exchange', 'Wearable'
]


def get_dls(params, test_only=False):
    """
    Build and return dataset-backed dataloaders for the dataset specified
    in `params.dset`.

    Parameters
    ----------
    params : object
        Any object with the required attributes, such as:
        - dset
        - context_points
        - target_points
        - batch_size
        - num_workers
        - features
        Optionally:
        - use_time_features

    Returns
    -------
    dls : DataLoaders
        A dataloader bundle containing train / valid / test loaders, along
        with metadata added at the end:
        - dls.vars : number of input variables / channels
        - dls.len  : input sequence length (context window)
        - dls.c    : target dimension
    """

    # --------------------------------------------------------------
    # Make sure the requested dataset is one of the supported names.
    # If not, raise a clear error.
    # --------------------------------------------------------------
    assert params.dset in DSETS, (
        f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    )

    # --------------------------------------------------------------
    # Some experiments may include calendar/time features.
    # If the parameter object does not define this flag, default to
    # False so downstream code does not break.
    # --------------------------------------------------------------
    if not hasattr(params, 'use_time_features'):
        params.use_time_features = False

    # ==============================================================
    # Dataset case 1: ETTm1
    # Minute-level Electricity Transformer Temperature dataset
    # --------------------------------------------------------------
    # size = [input_length, label_length, prediction_length]
    # Here label_length is set to 0 because this helper appears to be
    # used in a simplified forecasting/self-supervised setup.
    # ==============================================================
    if params.dset == 'ettm1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_ETT_minute,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # ==============================================================
    # Dataset case 2: ETTm2
    # Another minute-level ETT benchmark
    # ==============================================================
    elif params.dset == 'ettm2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_ETT_minute,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # ==============================================================
    # Dataset case 3: ETTh1
    # Hourly ETT benchmark
    # ==============================================================
    elif params.dset == 'etth1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_ETT_hour,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # ==============================================================
    # Dataset case 4: ETTh2
    # Another hourly ETT benchmark
    # ==============================================================
    elif params.dset == 'etth2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_ETT_hour,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # ==============================================================
    # Dataset case 5: Electricity
    # Uses Dataset_Custom because it is not one of the special ETT
    # dataset classes.
    # ==============================================================
    elif params.dset == 'electricity':
        root_path = '/data/datasets/public/electricity/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # ==============================================================
    # Dataset case 6: Traffic
    # ==============================================================
    elif params.dset == 'traffic':
        root_path = '/data/datasets/public/traffic/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # ==============================================================
    # Dataset case 7: Weather
    # ==============================================================
    elif params.dset == 'weather':
        root_path = '/data/datasets/public/weather/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # ==============================================================
    # Dataset case 8: Illness
    # This usually refers to the ILI benchmark used in forecasting
    # papers.
    # ==============================================================
    elif params.dset == 'illness':
        root_path = '/data/datasets/public/illness/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # ==============================================================
    # Dataset case 9: Exchange rate
    # ==============================================================
    elif params.dset == 'exchange':
        root_path = '/data/datasets/public/exchange_rate/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )
    
    
    
    #Custom Dataset 
    elif params.dset == 'Wearable':
        #root_path = './Homekit2020/data/processed/'
        
        import sys

        root = "/home/hice1/ezg6/projects/Homekit2020/src"
        if root not in sys.path:
            sys.path.insert(0, root)
        #print("This is system path", sys.path)
        import gc
        import time

        #from src.models.tasks import PredictFluPos
        
        from models.tasks import PredictFluPos 
        
        
        #root_path = './Homekit2020/data/processed/FullBypeople/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoadersV2(
            taskCls=PredictFluPos,
            task_kwargs={
                "train_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_userFull/train_7_day",
                "val_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_userFull/eval_7_day",
                "test_path": "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_userFull/test_7_day",
                "window_onset_min": 0,
                "window_onset_max": 0,
                #"shape": (10080, 8),

            },
             batch_size=params.batch_size,
             autoencoder_label=False,
             workers=params.num_workers,
             splits=('test',) if test_only else ('train', 'val', 'test'),
             label_filter=getattr(params, 'label_filter', 'all'),
             neg_subsample_ratio=getattr(params, 'neg_subsample_ratio', 0),
             seed=getattr(params, 'seed', 42),
        )

    # --------------------------------------------------------------
    # At this point, `dls` has been created for the requested dataset.
    #
    # The code below extracts some useful metadata from the first
    # training sample and stores it directly on the dataloader object.
    #
    # Assumption:
    #   dls.train.dataset[0] returns something like:
    #   (x, y)
    #
    # where:
    #   x.shape = [sequence_length, num_variables]
    #   y.shape = [target_dimension] or [prediction_length, ...]
    # --------------------------------------------------------------

    # Number of input variables / channels
    #dls.vars = dls.train.dataset[0][0].shape[1]

    # Save the context window length directly from params
    #dls.len = params.context_points
    #dls.len = dls.train.dataset[0][0].shape[0]

    # Save target dimensionality
    #dls.c = dls.train.dataset[0][1].shape[0]
    
    try:
    # Old dataset-style path — use whichever split is available
        _ref_dl = dls.train or dls.test or dls.valid
        sample = _ref_dl.dataset[0]

        if isinstance(sample, dict):
            sample_x = sample["inputs_embeds"]
            sample_y = sample["label"]
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            sample_x, sample_y = sample[0], sample[1]
        else:
            raise TypeError(f"Unsupported sample type from dataset[0]: {type(sample)}")

    except Exception:
        # New task/Petastorm-style path: inspect one batch instead
        batch = next(iter(dls.train))

        if isinstance(batch, dict):
            sample_x = batch["inputs_embeds"][0]
            sample_y = batch["label"][0:1]   # keep as 1D shape for compatibility
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            sample_x = batch[0][0]
            sample_y = batch[1][0:1]
        else:
            raise TypeError(f"Unsupported batch type from dls.train: {type(batch)}")

    # Infer dimensions
    dls.vars = sample_x.shape[1] if len(sample_x.shape) > 1 else 1
    dls.len = sample_x.shape[0]
    dls.c = sample_y.shape[0] if hasattr(sample_y, "shape") and len(sample_y.shape) > 0 else 1

    print("Finished DLS")
    return dls


if __name__ == "__main__":

    # --------------------------------------------------------------
    # Define a simple parameter container so the script can be run
    # directly for testing without argparse.
    # --------------------------------------------------------------
    class Params:
        dset = 'etth2'            # dataset name
        context_points = 384      # input sequence length
        target_points = 96        # forecast horizon
        batch_size = 64           # batch size
        num_workers = 8           # dataloader workers
        with_ray = False          # appears unused here, likely for other workflows
        features = 'M'            # multivariate mode

    # Model Params
    class Params:
        dset = 'Wearable'
        context_points = 10080
        target_points = 1
        batch_size = 32
        num_workers = 8
        with_ray = False
        features = 'M'
    # --------------------------------------------------------------
    # IMPORTANT:
    # Here the original code uses:
    #     params = Params
    # not
    #     params = Params()
    #
    # That means it is using the class itself as a parameter object,
    # relying on class attributes instead of instance attributes.
    #
    # This works because all fields are defined as class variables.
    # --------------------------------------------------------------
    params = Params

    # --------------------------------------------------------------
    # Build dataloaders for the chosen dataset configuration.
    # --------------------------------------------------------------
    dls = get_dls(params)

    # --------------------------------------------------------------
    # Iterate through the validation dataloader and print:
    # - batch index
    # - number of items in the batch tuple
    # - shape of inputs
    # - shape of targets
    #
    # This is mainly a sanity check to confirm the dataloader is
    # producing tensors with the expected dimensions.
    # --------------------------------------------------------------
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)

    # --------------------------------------------------------------
    # Drop into the debugger so you can inspect:
    # - dls
    # - dls.vars
    # - sample batches
    # - shapes
    # --------------------------------------------------------------
    breakpoint()