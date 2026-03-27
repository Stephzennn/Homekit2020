import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.data.timefeatures import time_features
import warnings


"""
class Dataset_HomeKitWearableV4(Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        size=None,
        features='M',
        data_path=["/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/train_7_day",
                   "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/eval_7_day", 
                "/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/test_7_day"],
        
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
        label = np.array([label], dtype=np.float32)   

        if self.use_time_features:
            return _torch(seq_x, label, seq_x_mark)
        else:
            return _torch(seq_x, label)

    def __len__(self):
        total = len(self.data_x) - self.seq_len + 1
        return max(0, (total + self.window_stride - 1) // self.window_stride)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


"""