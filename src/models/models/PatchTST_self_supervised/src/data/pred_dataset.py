import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.data.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

# This file defines several PyTorch Dataset classes for long-horizon time-series forecasting.
# Compared with the earlier version, these classes now use the keyword `split`
# instead of `flag`, and they also support an optional `use_time_features` switch
# to control whether time-mark covariates are returned from __getitem__.
#
# Core idea shared by all classes:
# --------------------------------
# Each dataset:
#   1. reads a CSV file
#   2. chooses train / val / test (or pred) rows
#   3. optionally scales the numeric data
#   4. builds time covariates from the timestamp column
#   5. returns rolling windows:
#        seq_x      -> input/history window
#        seq_y      -> decoder/history+forecast target window
#        seq_x_mark -> time features for seq_x
#        seq_y_mark -> time features for seq_y
#
# Important file-format expectation:
# ----------------------------------
# These classes expect a CSV-like table, not a parquet folder directly.
# The CSV should generally look like:
#
#   date, feature_1, feature_2, ..., feature_k, target
#
# or, for Dataset_Custom:
#
#   <time_col_name>, feature_1, feature_2, ..., feature_k, target
#
# Each row = one timestamp.
# Each non-time column = one variable/channel/modality.
#
# Example ETTh1 shape:
#   columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
#   rows   : ~17,420
#
# Example ETTm1 shape:
#   columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
#   rows   : ~69,680
#
# Example custom HomeKit-style shape if you want it to resemble ETTh1:
#   date, heart_rate, steps, hrv, sleep_flag, respiration
#
# Note:
# -----
# These datasets do NOT require user identity columns. If you include extra columns,
# they will be treated as ordinary features unless excluded upstream.
#
# Important output-shape conventions:
# -----------------------------------
# Let:
#   F = number of selected features
#   Tm = number of time-feature columns
#
# Then __getitem__(index) returns either:
#
#   if use_time_features == True:
#       (seq_x, seq_y, seq_x_mark, seq_y_mark)
#       seq_x      shape: [seq_len, F]
#       seq_y      shape: [label_len + pred_len, F]   (except Dataset_Pred)
#       seq_x_mark shape: [seq_len, Tm]
#       seq_y_mark shape: [label_len + pred_len, Tm]
#
#   if use_time_features == False:
#       (seq_x, seq_y)
#
# The helper _torch(...) at the end converts numpy arrays to float32 torch tensors.



class DataLoadersV1:
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
    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False
                 ):
        # root_path:
        #   Directory containing the CSV file.
        #
        # split:
        #   Which logical split this dataset should represent.
        #   Must be one of: 'train', 'val', 'test'
        #
        # size:
        #   [seq_len, label_len, pred_len]
        #   seq_len   = encoder/history length
        #   label_len = overlap/history length given to decoder
        #   pred_len  = forecast horizon
        #
        # features:
        #   'S'  -> only target column is used
        #   'M'  -> all columns except date are used
        #   'MS' -> treated like multivariate here at loading time
        #
        # data_path:
        #   CSV filename, e.g. ETTh1.csv
        #
        # target:
        #   Target column name. In ETT datasets this is usually 'OT'.
        #
        # scale:
        #   If True, StandardScaler is fit on the train split and applied to all data.
        #
        # timeenc:
        #   0 -> manual calendar extraction
        #   1 -> use time_features(...)
        #
        # freq:
        #   Frequency string for time_features. 'h' means hourly.
        #
        # use_time_features:
        #   If True, __getitem__ returns time-mark arrays too.
        #   If False, only (seq_x, seq_y) are returned.
        #
        # If no explicit size is provided:
        #   seq_len   = 24*4*4 = 384
        #   label_len = 24*4   = 96
        #   pred_len  = 24*4   = 96
        #
        # For hourly data:
        #   384 hours = 16 days
        #   96 hours  = 4 days
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # Only train/val/test splits are supported here.
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        # Store configuration.
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        # File location.
        # Expected file:
        #   root_path/data_path
        #
        # Example:
        #   /some/path/ETTh1.csv
        #
        # Expected CSV structure:
        #   date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # StandardScaler will fit mean/std on train split only.
        self.scaler = StandardScaler()

        # Read the hourly ETT CSV.
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # Hard-coded split boundaries used in the ETT benchmark.
        #
        # Assumes:
        #   12 months train
        #   4 months validation
        #   4 months test
        # with each month approximated as 30 days.
        #
        # Since this is hourly data:
        #   1 day = 24 rows
        #
        # border1 for val/test is shifted left by seq_len so that the first
        # validation/test sample still has enough preceding history to form seq_x.
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Choose which feature columns to model.
        #
        # df_raw.columns[1:] means "all columns except the first date column".
        #
        # If multivariate:
        #   df_data shape = [N, number_of_signal_columns]
        #
        # If single-variate:
        #   df_data shape = [N, 1]
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Fit scaler on train split only, then transform the full selected data.
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract the timestamp column only for the current split.
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            # Manual calendar covariates for hourly data:
            #   month, day, weekday, hour
            #
            # Resulting data_stamp shape:
            #   [split_rows, 4]
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # Use external time feature encoder.
            # Usually returns [num_time_features, split_rows], so transpose to
            # [split_rows, num_time_features].
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # Save the selected split.
        #
        # self.data_x and self.data_y are identical base arrays here.
        # The distinction is introduced later by how windows are sliced.
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Sliding window logic:
        #   seq_x covers [index : index + seq_len]
        #   seq_y covers the decoder overlap + prediction horizon
        s_begin = index
        s_end = s_begin + self.seq_len

        # Decoder starts label_len steps before end of encoder history.
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Main numeric windows.
        seq_x = self.data_x[s_begin:s_end]         # shape [seq_len, F]
        seq_y = self.data_y[r_begin:r_end]         # shape [label_len + pred_len, F]

        # Time-feature windows.
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Optionally return time features depending on caller preference.
        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        # Number of valid sliding windows in this split.
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # Convert scaled values back to original scale.
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_time_features=False
                 ):
        # Same logic as Dataset_ETT_hour, but for minute-resolution ETT data.
        #
        # In the ETT benchmark, the "minute" datasets are typically 15-minute sampled.
        # That is why split boundaries below are multiplied by 4.
        #
        # Default size values are numerically the same as above:
        #   seq_len = 384
        #   label_len = 96
        #   pred_len = 96
        #
        # But now one time step = 15 minutes, so:
        #   384 steps = 96 hours = 4 days
        #   96 steps  = 24 hours = 1 day
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        # Expected CSV similar to ETTh1 but with many more rows.
        # Typical file:
        #   ETTm1.csv
        # Typical shape:
        #   ~69,680 rows, 1 date column + 7 signal columns
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Read minute-level CSV.
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # Hard-coded ETT minute split protocol.
        # Each hour has 4 measurements (15-minute interval), so multiply by 4.
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select feature columns.
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Fit scaler on train partition only.
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract timestamps for the current split.
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            # Manual time features for minute-level data:
            #   month, day, weekday, hour, minute_bucket
            #
            # minute_bucket = minute // 15, which maps:
            #   00 -> 0
            #   15 -> 1
            #   30 -> 2
            #   45 -> 3
            #
            # This strongly assumes the data are sampled every 15 minutes.
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Same sliding-window extraction as Dataset_ETT_hour.
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2
                 ):
        # This is the flexible/general dataset loader intended for arbitrary CSVs.
        #
        # Compared with the ETT-specific loaders:
        # - split boundaries are NOT hard-coded to calendar months
        # - time column name is configurable
        # - train/test ratios are configurable
        #
        # This is the class most suitable for adapting a custom dataset like:
        #   date, heart_rate, steps, hrv, sleep_state, respiration, target
        #
        # Important expected CSV structure:
        #   [time_col_name, other feature columns, target column]
        #
        #
        # and you want multivariate forecasting, then:
        #   features='M'
        #   target='sleep' (for example)
        #
        # Default train_split=0.7 and test_split=0.2 imply:
        #   validation split = remaining 0.1
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # User-controlled split ratios.
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Read custom CSV.
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        # This block gets all columns, but note that the earlier code that explicitly
        # reordered columns is commented out.
        #
        # That means this loader currently assumes the input CSV is already arranged
        # appropriately, or at least that "all non-first columns" are valid data columns.
        #
        # If your CSV has extra metadata columns after the time column, they will be
        # treated as ordinary features in multivariate mode.
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        # Compute split sizes based on row count.
        # Example:
        #   if N = 100000 rows,
        #   num_train = 70000
        #   num_test  = 20000
        #   num_vali  = 10000
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test

        # As before, validation and test start positions are shifted left by seq_len
        # so that the first sample in each split can still access enough history.
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Feature selection.
        #
        # Caution:
        # For multivariate mode, this code uses df_raw.columns[1:], which means:
        #   "every column except the first one"
        #
        # So it assumes the first column is the time column.
        # If your CSV has:
        #   date, user_id, hr, steps, ...
        # then user_id would also be treated as a feature unless removed upstream.
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Fit scaler on train slice only.
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract time column for current split.
        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])

        if self.timeenc == 0:
            # Manual time features:
            #   month, day, weekday, hour
            #
            # This is fine for hourly or finer-resolution timestamps.
            # If your data are minute-level and you need minute buckets too,
            # this class as written does NOT currently add them.
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            # Generic external time-feature encoding based on freq.
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Sliding-window extraction for custom data.
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_HomeKitWearableV2(Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        size=None,
        features='M',
        data_path=['WearableTrain.csv', 'WearableEval.csv', 'WearableTest.csv'],
        target='InfectionStatus',
        scale=True,
        timeenc=0,
        freq='h',
        time_col_name='date',
        use_time_features=False,
        train_split=0.7,
        test_split=0.2,
        window_stride=60
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert split in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.window_stride = window_stride

        # Kept for compatibility, even though pre-split files make these unnecessary
        self.train_split = train_split
        self.test_split = test_split

        self.root_path = root_path
        self.data_path_train = data_path[0]
        self.data_path_eval = data_path[1]
        self.data_path_test = data_path[2]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # ----------------------------------------------------------
        # Read the already-separated train / eval / test CSV files
        # ----------------------------------------------------------
        df_rawTrain = pd.read_csv(os.path.join(self.root_path, self.data_path_train))
        df_rawEval = pd.read_csv(os.path.join(self.root_path, self.data_path_eval))
        df_rawTest = pd.read_csv(os.path.join(self.root_path, self.data_path_test))

        # ----------------------------------------------------------
        # Select the dataframe for the current split
        # ----------------------------------------------------------
        if self.set_type == 0:
            df_raw = df_rawTrain
        elif self.set_type == 1:
            df_raw = df_rawEval
        else:
            df_raw = df_rawTest

        # ----------------------------------------------------------
        # Explicit feature columns for the wearable dataset
        # Excludes the time column and the target column
        # ----------------------------------------------------------
        feature_cols = [
            'steps',
            'heart_rate',
            'missing_heartrate',
            'missing_steps',
            'sleep_classic_0',
            'sleep_classic_1',
            'sleep_classic_2',
            'sleep_classic_3'
        ]

        # ----------------------------------------------------------
        # Build feature matrix for the current split
        # and training feature matrix for scaler fitting
        # ----------------------------------------------------------
        if self.features in ['M', 'MS']:
            df_data = df_raw[feature_cols]
            df_train_data = df_rawTrain[feature_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_train_data = df_rawTrain[[self.target]]
        else:
            raise ValueError(f"Unsupported features setting: {self.features}")

        # ----------------------------------------------------------
        # Fit scaler on TRAIN ONLY, then transform current split
        # ----------------------------------------------------------
        if self.scale:
            self.scaler.fit(df_train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ----------------------------------------------------------
        # Build time features for the current split only
        # ----------------------------------------------------------
        df_stamp = df_raw[[self.time_col_name]].copy()
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda x: x.month)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda x: x.day)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda x: x.weekday())
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda x: x.hour)
            df_stamp['minute'] = df_stamp[self.time_col_name].apply(lambda x: x.minute)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[self.time_col_name].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc setting: {self.timeenc}")

        # ----------------------------------------------------------
        # Keep PatchTST-style behavior for now:
        # input and target both come from the same data matrix
        # ----------------------------------------------------------
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index * self.window_stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features:
            return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else:
            return _torch(seq_x, seq_y)

    def __len__(self):
        total = len(self.data_x) - self.seq_len - self.pred_len + 1
        return max(0, (total + self.window_stride - 1) // self.window_stride)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
class Dataset_HomeKitWearable(Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        size=None,
        features='M',
        data_path=['WearableTrain.csv', 'WearableEval.csv', 'WearableTest.csv'],
        target='InfectionStatus',
        scale=True,
        timeenc=0,
        freq='h',
        time_col_name='date',
        use_time_features=False,
        train_split=0.7,
        test_split=0.2
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert split in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # Kept for compatibility, even though pre-split files make these unnecessary
        self.train_split = train_split
        self.test_split = test_split

        self.root_path = root_path
        self.data_path_train = data_path[0]
        self.data_path_eval = data_path[1]
        self.data_path_test = data_path[2]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # ----------------------------------------------------------
        # Read the already-separated train / eval / test CSV files
        # ----------------------------------------------------------
        df_rawTrain = pd.read_csv(os.path.join(self.root_path, self.data_path_train))
        df_rawEval = pd.read_csv(os.path.join(self.root_path, self.data_path_eval))
        df_rawTest = pd.read_csv(os.path.join(self.root_path, self.data_path_test))

        # ----------------------------------------------------------
        # Select the dataframe for the current split
        # ----------------------------------------------------------
        if self.set_type == 0:
            df_raw = df_rawTrain
        elif self.set_type == 1:
            df_raw = df_rawEval
        else:
            df_raw = df_rawTest

        # ----------------------------------------------------------
        # Explicit feature columns for the wearable dataset
        # Excludes the time column and the target column
        # ----------------------------------------------------------
        feature_cols = [
            'steps',
            'heart_rate',
            'missing_heartrate',
            'missing_steps',
            'sleep_classic_0',
            'sleep_classic_1',
            'sleep_classic_2',
            'sleep_classic_3'
        ]

        # ----------------------------------------------------------
        # Build feature matrix for the current split
        # and training feature matrix for scaler fitting
        # ----------------------------------------------------------
        if self.features in ['M', 'MS']:
            df_data = df_raw[feature_cols]
            df_train_data = df_rawTrain[feature_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_train_data = df_rawTrain[[self.target]]
        else:
            raise ValueError(f"Unsupported features setting: {self.features}")

        # ----------------------------------------------------------
        # Fit scaler on TRAIN ONLY, then transform current split
        # ----------------------------------------------------------
        if self.scale:
            self.scaler.fit(df_train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ----------------------------------------------------------
        # Build time features for the current split only
        # ----------------------------------------------------------
        df_stamp = df_raw[[self.time_col_name]].copy()
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda x: x.month)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda x: x.day)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda x: x.weekday())
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda x: x.hour)
            df_stamp['minute'] = df_stamp[self.time_col_name].apply(lambda x: x.minute)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[self.time_col_name].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc setting: {self.timeenc}")

        # ----------------------------------------------------------
        # Keep PatchTST-style behavior for now:
        # input and target both come from the same data matrix
        # ----------------------------------------------------------
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features:
            return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else:
            return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_HomeKitWearableV3(Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        size=None,
        features='M',
        data_path=['WearableTrain.csv', 'WearableEval.csv', 'WearableTest.csv'],
        target='InfectionStatus',
        scale=True,
        timeenc=0,
        freq='h',
        time_col_name='date',
        use_time_features=False,
        train_split=0.7,
        test_split=0.2,
        window_stride=1440
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert split in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.window_stride = window_stride

        self.train_split = train_split
        self.test_split = test_split

        self.root_path = root_path
        self.data_path_train = data_path[0]
        self.data_path_eval = data_path[1]
        self.data_path_test = data_path[2]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Load only the split we actually need
        if self.set_type == 0:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path_train))
        elif self.set_type == 1:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path_eval))
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path_test))
        
        print("After the df_raw")
        feature_cols = [
            'steps',
            'heart_rate',
            'missing_heartrate',
            'missing_steps',
            'sleep_classic_0',
            'sleep_classic_1',
            'sleep_classic_2',
            'sleep_classic_3'
        ]

        if self.features in ['M', 'MS']:
            df_data = df_raw[feature_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features setting: {self.features}")

        
        print("self.features in ['M', 'MS']:")
        # Fit scaler only on train split, if scaling is enabled
        if self.scale:
            df_train = pd.read_csv(os.path.join(self.root_path, self.data_path_train))

            if self.features in ['M', 'MS']:
                df_train_data = df_train[feature_cols]
            elif self.features == 'S':
                df_train_data = df_train[[self.target]]
            else:
                raise ValueError(f"Unsupported features setting: {self.features}")

            self.scaler.fit(df_train_data.values)
            data = self.scaler.transform(df_data.values)

            del df_train
            del df_train_data
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col_name]].copy()
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda x: x.month)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda x: x.day)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda x: x.weekday())
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda x: x.hour)
            df_stamp['minute'] = df_stamp[self.time_col_name].apply(lambda x: x.minute)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[self.time_col_name].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc setting: {self.timeenc}")

        # Inputs
        self.data_x = data
        self.data_stamp = data_stamp

        # Classification labels
        self.labels = df_raw[self.target].values.astype(np.float32)

    def __getitem__(self, index):
        s_begin = index * self.window_stride
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        # single binary label for this window
        label = self.labels[s_end - 1]
        label = np.array([label], dtype=np.float32)   # shape [1]

        if self.use_time_features:
            return _torch(seq_x, label, seq_x_mark)
        else:
            return _torch(seq_x, label)

    def __len__(self):
        total = len(self.data_x) - self.seq_len + 1
        return max(0, (total + self.window_stride - 1) // self.window_stride)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, split='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # This dataset is for prediction/inference mode, not train/val/test learning.
        #
        # It uses the LAST seq_len rows as observed history and creates future timestamps
        # for pred_len steps ahead.
        #
        # split must be 'pred' only.
        #
        # inverse:
        #   If True, self.data_y stores original-scale values.
        #   If False, self.data_y stores scaled values.
        #
        # cols:
        #   Optional explicit list of feature columns to include.
        #
        # Expected CSV structure:
        #   date, feature_1, ..., feature_k, target
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert split in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Read the full CSV.
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        # Reorder columns so target is placed last and date first.
        # If cols is provided, only those columns are used (plus target).
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')

        # Final ordering:
        #   date, feature_1, ..., feature_k, target
        df_raw = df_raw[['date'] + cols + [self.target]]

        # For prediction, only the last seq_len rows are used as model input history.
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        # Select multivariate or single target.
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # In prediction mode the scaler is fit on the full available df_data.
        # This differs from the train/val/test datasets, where it is fit only on train.
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract the observed tail timestamps.
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)

        # Create future timestamps from the last observed date.
        # periods=self.pred_len+1 creates one extra first point equal to the last observed one.
        # Later pred_dates[1:] removes that duplication.
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        # Build a full time table consisting of:
        #   observed tail timestamps + future timestamps
        # Total length:
        #   seq_len + pred_len
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])

        if self.timeenc == 0:
            # Manual time features including minute bucket because default freq is 15min.
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # History window only.
        self.data_x = data[border1:border2]

        # If inverse=True, keep original-scale y data for the observed history.
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

        # Time features for both history and future forecast horizon.
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Similar window logic, but in prediction mode there is usually only one valid sample.
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Encoder/history input.
        seq_x = self.data_x[s_begin:s_end]

        # In prediction mode, seq_y only includes known decoder-history portion,
        # not the unknown future target values.
        #
        # Shape:
        #   [label_len, F]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]

        # Time features:
        # seq_x_mark -> history timestamps
        # seq_y_mark -> decoder-history + future timestamps
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # Usually 1 when len(self.data_x) == seq_len.
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def _torch(*dfs):
    # Helper function:
    #   takes any number of numpy arrays
    #   converts each to float32 torch tensors
    #   returns them as a tuple
    #
    # Example:
    #   _torch(seq_x, seq_y)
    # returns:
    #   (torch.FloatTensor(seq_x), torch.FloatTensor(seq_y))
    return tuple(torch.from_numpy(x).float() for x in dfs)




"""
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_time_features=False
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, split='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def _torch(*dfs):
    return tuple(torch.from_numpy(x).float() for x in dfs)

"""
