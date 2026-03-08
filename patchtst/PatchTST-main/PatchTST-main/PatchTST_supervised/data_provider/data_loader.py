import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler



from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

"""
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

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
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
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
            # print(self.scaler.mean_)
            # exit()
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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
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
        assert flag in ['pred']

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

"""

# This file defines several PyTorch Dataset classes for time-series forecasting.
# All of them produce sliding windows of the form:
#   seq_x      : encoder input/history
#   seq_y      : decoder target/history+future
#   seq_x_mark : time features aligned with seq_x
#   seq_y_mark : time features aligned with seq_y
#
# Expected returned shapes from __getitem__:
#   seq_x      -> [seq_len, num_features]
#   seq_y      -> [label_len + pred_len, num_features]   # except Dataset_Pred, see below
#   seq_x_mark -> [seq_len, num_time_features]
#   seq_y_mark -> [label_len + pred_len, num_time_features]
#
# Broad assumptions shared by these classes:
# 1. The CSV file must contain a 'date' column.
# 2. For multivariate mode ('M' or 'MS'), every column except 'date' is treated as a signal column.
# 3. For univariate mode ('S'), only the column named by `target` is used.
# 4. Data are converted into overlapping windows using a rolling/sliding-index strategy.
# 5. StandardScaler is fit only on the training slice (except Dataset_Pred, where it is fit on all available rows).
#
# Typical CSV layout expected by these loaders:
#   date, feature_1, feature_2, ..., feature_k, target
#
# Example shapes:
# - If df_raw has N rows and k usable features:
#     data      -> [N, k] after extraction/scaling
#     data_x    -> [split_length, k]
#     data_stamp-> [split_length, num_time_features]
#
# Important semantic note on `features`:
# - 'S'  : single-variate input/output using only target column
# - 'M'  : multivariate input/output using all columns except date
# - 'MS' : multivariate input, single target in some downstream settings;
#          in this file, it is grouped with 'M' at the data loading stage


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size should be [seq_len, label_len, pred_len]
        # seq_len   = how much history the encoder sees
        # label_len = how much recent history is passed into decoder side
        # pred_len  = how far ahead to forecast
        #
        # If no explicit size is provided, defaults are used:
        #   seq_len   = 24*4*4 = 384
        #   label_len = 24*4   = 96
        #   pred_len  = 24*4   = 96
        #
        # For hourly ETT data, 384 steps = 16 days of hourly history.
        # 96 steps = 4 days.
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # flag determines which split this dataset instance will represent.
        # Only train/test/val are allowed.
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # features controls whether this is single-variate or multivariate loading.
        # target is the target column name, e.g. 'OT' in standard ETT datasets.
        # scale=True means StandardScaler will normalize values using training split stats.
        # timeenc:
        #   0 -> manually build calendar/time fields (month/day/weekday/hour)
        #   1 -> use external `time_features(...)` utility
        # freq='h' says the timestamps are hourly.
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # root_path is directory holding CSV.
        # data_path is file name, e.g. ETTh1.csv.
        #
        # Expected ETTh1/ETTh2 file:
        #   - must be a CSV
        #   - must include a 'date' column
        #   - usually includes columns like HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        #   - shape is typically [~17420 rows, 8 columns] including 'date'
        #     for standard benchmark versions
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # StandardScaler stores mean/std and transforms values feature-wise.
        self.scaler = StandardScaler()

        # Read the CSV into a pandas DataFrame.
        # Expected shape for canonical ETTh1:
        #   rows ≈ 17420
        #   cols = 1 date column + 7 signal columns
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # These borders implement the standard ETT split protocol.
        # For hourly data:
        #   12 months train
        #   4 months val
        #   4 months test
        #
        # Since they approximate each month as 30 days:
        #   train rows = 12 * 30 * 24
        #   val rows   =  4 * 30 * 24
        #   test rows  =  4 * 30 * 24
        #
        # border1 for val/test is shifted left by seq_len so that the first window
        # in val/test still has enough history to build seq_x.
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select which data columns will be modeled.
        # df_raw.columns[1:] means: all columns except 'date'.
        #
        # If features in {'M','MS'}:
        #   use all signal columns
        #   resulting df_data shape: [N, num_signals]
        #
        # If features == 'S':
        #   use only target column
        #   resulting df_data shape: [N, 1]
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Fit scaler only on training slice, then transform all rows.
        # This is correct practice: no leakage from val/test statistics.
        #
        # train_data shape:
        #   [train_rows, num_selected_features]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            # If scale=False, use raw numeric values directly.
            data = df_data.values

        # Extract dates only for the current split.
        # df_stamp shape before adding derived fields:
        #   [split_rows, 1]
        df_stamp = df_raw[['date']][border1:border2]

        # Convert textual date strings to pandas datetime objects.
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            # Manual calendar encoding.
            # For hourly data, time markers are:
            #   month, day, weekday, hour
            # Final data_stamp shape:
            #   [split_rows, 4]
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # External time encoding.
            # Depending on implementation of time_features(...),
            # the returned shape is typically [num_time_features, split_rows],
            # so transpose makes it [split_rows, num_time_features].
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # data_x and data_y are identical here because this loader supports forecasting:
        # the model gets history from data_x and predicts future values from data_y windows.
        #
        # Shapes:
        #   self.data_x -> [split_rows, num_selected_features]
        #   self.data_y -> [split_rows, num_selected_features]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # Time features aligned with the same split range.
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Build one sliding window sample.
        #
        # Encoder/history window:
        #   [s_begin : s_end] with length seq_len
        s_begin = index
        s_end = s_begin + self.seq_len

        # Decoder window starts label_len steps before end of encoder history.
        # This lets decoder receive some recent observed points plus future horizon.
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x shape: [seq_len, num_features]
        seq_x = self.data_x[s_begin:s_end]

        # seq_y shape: [label_len + pred_len, num_features]
        # It contains recent overlap + future target horizon.
        seq_y = self.data_y[r_begin:r_end]

        # Aligned time features.
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # Number of valid sliding windows in the current split.
        # Need enough room for both seq_len history and pred_len future.
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # Convert scaled values back to original scale using fitted training scaler.
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # Same semantics as Dataset_ETT_hour, but for minute-level ETT data.
        # Canonical ETT minute data are usually at 15-minute intervals.
        #
        # Default lengths numerically match the hourly version, but here each step
        # is one 15-minute interval, so:
        #   seq_len = 384  -> 384 * 15 min = 96 hours = 4 days
        #   label_len = 96 -> 1 day
        #   pred_len = 96  -> 1 day
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # freq='t' denotes minutely frequency in pandas offset aliases.
        # In the benchmark setting, ETT minute data are typically 15-minute sampled.
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # Expected CSV such as ETTm1.csv:
        #   - must contain 'date'
        #   - typically same signal columns as ETTh*
        #   - standard benchmark rows ≈ 69680
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Read minute-level CSV.
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # Split protocol for minute-level data.
        # Multiplied by 4 because there are 4 samples/hour at 15-minute resolution.
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Same feature-selection logic as above.
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Fit scaler on training portion only.
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract date slice for current split.
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            # Manual time features for minute-level data:
            #   month, day, weekday, hour, minute_bucket
            #
            # The code maps minute -> minute // 15, which compresses minute values
            # into 4 quarter-hour buckets:
            #   0, 15, 30, 45  -> 0,1,2,3
            #
            # This strongly suggests the expected CSV is sampled at 15-minute resolution.
            # Final shape:
            #   [split_rows, 5]
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # Current split arrays.
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Same sliding-window logic as hourly version.
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # This class generalizes the same logic to arbitrary custom CSV files.
        # Unlike ETT classes, it does not use hard-coded calendar split lengths.
        # Instead, it uses 70% train / 10% val / 20% test.
        #
        # The CSV still must contain:
        #   - 'date' column
        #   - target column named by `target`
        #   - any number of additional feature columns
        #
        # Expected generic shape:
        #   [N rows, 1 + p columns]
        # where 1 column is 'date' and remaining p-1/ p are features depending on target inclusion.
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Read user-provided custom CSV.
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # Reorder columns so that:
        #   ['date'] + non-target features + [target]
        #
        # This is useful because some downstream implementations assume the target
        # is the last column.
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # Split into 70% train, 10% val, 20% test.
        # Note: because num_vali is "whatever remains" after int truncation of train/test,
        # the exact val fraction may not be precisely 10% for every N.
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # border1 for val/test again shifts left by seq_len to preserve enough history.
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select either all features or target only.
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Fit on train split only, transform all.
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Build time markers for current split.
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            # For custom data with default freq='h', this generates month/day/weekday/hour.
            # If your custom data are minute-level or daily-level, this encoding may be incomplete
            # unless you adjust freq/timeenc logic elsewhere.
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # Save current split arrays.
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Standard rolling window extraction.
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # Number of windows available in this split.
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # This class is for prediction/inference rather than train/val/test splitting.
        #
        # It takes the last seq_len rows of the CSV as history and constructs future timestamps
        # for pred_len steps ahead.
        #
        # Expected use:
        #   - you already have a trained model
        #   - you want one final prediction window from the tail of the dataset
        #
        # `inverse=True` controls whether self.data_y stores raw unscaled values.
        # This can be useful if you want decoder history in original scale for certain pipelines.
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['pred']

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

        # Read full CSV.
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        # If a custom explicit list of columns is given, use it (excluding target first,
        # then append target at the end). Otherwise infer all columns.
        #
        # Final ordering:
        #   ['date'] + chosen non-target columns + [target]
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # For prediction mode, only the last seq_len rows are used as observed history.
        # border1:border2 defines that last window.
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        # Feature selection as before.
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Important difference from train/val/test datasets:
        # here scaler is fit on the whole available df_data, not just a train partition,
        # because this dataset is intended only for final prediction use.
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract the observed tail dates.
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)

        # Construct future timestamps for prediction horizon.
        # pd.date_range starts from the last observed timestamp, so periods=pred_len+1
        # gives one extra initial date; [1:] later discards that duplicate start.
        #
        # If freq='15min' and pred_len=96, this creates 96 future timestamps covering 24 hours.
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        # Build one timestamp table containing:
        #   observed tail timestamps + future prediction timestamps
        # Total length = seq_len + pred_len
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])

        if self.timeenc == 0:
            # Manual date features.
            # Since default freq here is '15min', minute bucket is included and quarter-hour compressed.
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

        # data_x is the final observed history window only: shape [seq_len, num_features]
        self.data_x = data[border1:border2]

        # data_y depends on inverse flag:
        #   inverse=False -> scaled values
        #   inverse=True  -> raw original values
        #
        # Still only the observed history region is stored here.
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

        # data_stamp now covers both observed history and future horizon:
        # shape [seq_len + pred_len, num_time_features]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # In prediction mode, indexing still supports windowing over self.data_x,
        # but since self.data_x usually has exactly seq_len rows, __len__ is often 1.
        s_begin = index
        s_end = s_begin + self.seq_len

        # Decoder begins label_len steps before end of history.
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # History input.
        seq_x = self.data_x[s_begin:s_end]

        # For prediction mode, seq_y only includes known decoder-context portion,
        # not the unknown future values.
        #
        # Shape:
        #   [label_len, num_features]
        #
        # This differs from the train/val/test datasets, where seq_y includes
        # label_len + pred_len.
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]

        # seq_x_mark covers encoder/history timestamps.
        seq_x_mark = self.data_stamp[s_begin:s_end]

        # seq_y_mark covers both decoder-known history + future prediction timestamps.
        # Shape:
        #   [label_len + pred_len, num_time_features]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # Usually 1 if len(self.data_x) == seq_len.
        # More generally allows rolling prediction windows if data_x is longer.
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        # Convert model outputs back to original units.
        return self.scaler.inverse_transform(data)