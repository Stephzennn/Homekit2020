import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
"""
class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int=0,
        collate_fn=None,
        shuffle_train = True,
        shuffle_val = False
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val
    
        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()        
 
        
    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):        
        return self._make_dloader("val", shuffle=self.shuffle_val)

    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if len(dataset) == 0: return None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # check of test_data is already a DataLoader
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader): 
            return test_data

        # get batch_size if not defined
        if batch_size is None: batch_size=self.batch_size        
        # check if test_data is Dataset, if not, wrap Dataset
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)        
        
        # create a new DataLoader from Dataset 
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data

"""

# This file defines a lightweight wrapper class around PyTorch DataLoader creation.
# Its goal is to centralize how train/validation/test dataloaders are built from
# a dataset class.
#
# High-level idea:
# - You pass in a dataset class (datasetCls), not a dataset instance.
# - You also pass keyword arguments needed to construct that dataset class.
# - This wrapper automatically creates:
#     self.train
#     self.valid
#     self.test
#   by calling the dataset class three times with split="train"/"val"/"test".
#
# Important conceptual requirement:
# --------------------------------
# The dataset class you pass in MUST support a constructor like:
#
#     datasetCls(..., split="train")
#
# or more generally accept a keyword argument named `split`.
#
# That means this DataLoaders class assumes your dataset class knows how to:
# - read the right underlying data for a given split, or
# - partition one dataset into train/val/test internally.
#
# This file itself does NOT read CSV files, parquet files, or tensors directly.
# It delegates all actual data-reading logic to `datasetCls`.
#
# So if the dataset class needs:
# - a CSV path
# - a parquet folder
# - a root directory
# - metadata files
# - a target column name
# - sequence length
# - transforms
# then those must be passed through `dataset_kwargs`.
#
# Example usage:
# --------------
# Suppose you have a dataset class like:
#
#   class MyDataset(Dataset):
#       def __init__(self, root_path, split, seq_len, target):
#           ...
#
# Then you might build loaders as:
#
#   dls = DataLoaders(
#       datasetCls=MyDataset,
#       dataset_kwargs={
#           "root_path": "/path/to/data.csv",
#           "seq_len": 512,
#           "target": "heart_rate",
#       },
#       batch_size=64,
#       workers=4
#   )
#
# Internally, this wrapper will create:
#   MyDataset(root_path=..., seq_len=..., target=..., split="train")
#   MyDataset(root_path=..., seq_len=..., target=..., split="val")
#   MyDataset(root_path=..., seq_len=..., target=..., split="test")
#
# and then wrap each one in a torch.utils.data.DataLoader.


import warnings

import torch
from torch.utils.data import DataLoader


class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int=0,
        collate_fn=None,
        shuffle_train = True,
        shuffle_val = False
    ):
        # datasetCls:
        #   A DATASET CLASS, not an already-instantiated dataset object.
        #   Usually something that subclasses torch.utils.data.Dataset.
        #
        # dataset_kwargs:
        #   Dictionary of keyword arguments used to instantiate datasetCls.
        #   Example:
        #       {
        #           "root_path": "...",
        #           "seq_len": 512,
        #           "target": "OT"
        #       }
        #
        # batch_size:
        #   Number of samples per batch returned by the DataLoader.
        #
        # workers:
        #   Number of subprocesses used by PyTorch DataLoader to load batches in parallel.
        #   If 0, everything loads in the main process.
        #
        # collate_fn:
        #   Optional custom collate function.
        #   Use this if your dataset returns objects that need special batching logic.
        #
        # shuffle_train / shuffle_val:
        #   Controls whether train/validation datasets are shuffled at the DataLoader level.
        #
        # This constructor builds all three dataloaders immediately:
        #   self.train
        #   self.valid
        #   self.test
        super().__init__()

        # Save the dataset class itself for later use.
        self.datasetCls = datasetCls

        # Save batch size globally for this wrapper instance.
        self.batch_size = batch_size
        
        # This block removes "split" if the caller already included it in dataset_kwargs.
        #
        # Why?
        # Because this wrapper wants to control split automatically in _make_dloader().
        # If the caller passed split manually, it could conflict with the wrapper's logic.
        #
        # Example:
        #   dataset_kwargs = {"root_path": "...", "split": "train"}
        #
        # This would be problematic because then later:
        #   datasetCls(..., split="val")
        # would duplicate split definitions.
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]

        # Save all dataset construction parameters except split.
        self.dataset_kwargs = dataset_kwargs

        # Number of worker processes the DataLoader should use.
        self.workers = workers

        # Optional batching function.
        self.collate_fn = collate_fn

        # Store shuffle behavior for train/val loaders.
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val
    
        # Build train, validation, and test dataloaders immediately.
        #
        # Important:
        # These attributes hold DataLoader objects (or None if dataset length is zero).
        #
        # So after construction:
        #   dls.train  -> DataLoader for split='train'
        #   dls.valid  -> DataLoader for split='val'
        #   dls.test   -> DataLoader for split='test'
        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()        
 
        
    def train_dataloader(self):
        # Build the training DataLoader.
        # Usually shuffling is enabled for better SGD behavior.
        return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):        
        # Build the validation DataLoader.
        # Usually shuffling is disabled, though caller can override.
        return self._make_dloader("val", shuffle=self.shuffle_val)

    def test_dataloader(self):
        # Build the test DataLoader.
        # By convention test data are not shuffled.
        return self._make_dloader("test", shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        # Core helper that constructs one dataset instance and wraps it in a DataLoader.
        #
        # It does:
        #   1. Instantiate datasetCls(..., split=split)
        #   2. If dataset is empty, return None
        #   3. Otherwise wrap it in torch.utils.data.DataLoader
        #
        # This means your dataset class must accept the keyword argument `split`.
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)

        # If the dataset has no samples, do not create a DataLoader.
        # Returning None prevents downstream crashes from empty datasets.
        #
        # So if:
        #   len(dataset) == 0
        # then:
        #   self.train / self.valid / self.test may become None
        if len(dataset) == 0: return None

        # Wrap the dataset in a PyTorch DataLoader.
        #
        # The DataLoader handles:
        # - batching
        # - optional shuffling
        # - multi-worker loading
        # - optional custom collate logic
        #
        # Output batch shape depends entirely on what the dataset returns and how collate_fn works.
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    @classmethod
    def add_cli(self, parser):
        # Utility method for argparse-style command-line interfaces.
        #
        # It adds standard dataloader-related arguments to a parser.
        #
        # Example:
        #   parser = argparse.ArgumentParser()
        #   DataLoaders.add_cli(parser)
        #
        # Then command line can accept:
        #   --batch_size 128
        #   --workers 6
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # This method is meant to create or normalize an evaluation/test dataloader.
        #
        # It accepts either:
        # - a DataLoader already
        # - a Dataset
        # - raw data that can be converted into a Dataset via self.train.dataset.new(...)
        #
        # Then it returns a DataLoader.
        #
        # This is useful when you want to quickly evaluate on:
        # - a new dataset
        # - a custom tensor/array/table
        # - an already-prepared DataLoader

        # check if test_data is already a DataLoader
        # If it is, return it directly without wrapping again

        if isinstance(test_data, DataLoader):
            return test_data

        # If batch_size was not explicitly passed, reuse the wrapper's default batch size.
        if batch_size is None: batch_size=self.batch_size        

        # check if test_data is Dataset, if not, wrap Dataset
        #
        # This code assumes the training dataset object has a method called `.new(...)`
        # that knows how to convert arbitrary raw input into a new Dataset object.
        #
        # This is NOT part of the standard PyTorch Dataset API.
        # So for this to work, your custom dataset class likely needs to implement:
        #
        #   dataset.new(raw_data)
        #
        # If it does not, this line will fail.
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)        
        
        # create a new DataLoader from Dataset 
        #
        # This line assumes the training DataLoader object has a `.new(...)` method.
        # Standard torch DataLoader does NOT have a `.new(...)` method.
        #
        # So this code likely depends on a custom patched DataLoader wrapper used elsewhere
        # in the project.
        #
        # In a plain PyTorch setting, this line would normally not work unless:
        # - self.train is not a vanilla DataLoader, or
        # - the project monkey-patches/extends DataLoader functionality.
        #
        # Conceptually, the intent is:
        #   "take the dataset and create another dataloader with a possibly different batch size"
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data
    
    
"""

from torch.utils.data import DataLoader, Dataset





class DataLoadersV1:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int = 0,
        collate_fn=None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.dataset_kwargs = dict(dataset_kwargs)
        self.batch_size = batch_size
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        # Let this wrapper control split
        self.dataset_kwargs.pop("split", None)

    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._make_dloader("val", shuffle=self.shuffle_val)

    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)

    def get_dataloaders(self):
        #Return train, val, test dataloaders directly.
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        test_loader = self.test_dataloader()
        return train_loader, val_loader, test_loader

    def _make_dloader(self, split, shuffle=False):
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