class Dataset_HomeKitWearable(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path=['WearableTrain.csv','WearableEval.csv','WearableTest.csv'],
                 target='InfectionStatus', scale=True, timeenc=0, freq='h',
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
        #self.data_path = data_path
        self.data_path_train = data_path[0]
        self.data_path_eval = data_path[1]
        self.data_path_test = data_path[2]
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Read custom CSV.
        #df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                  self.data_path))

        df_rawTrain = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path_train))
        df_rawEval = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path_eval))
        df_rawTest = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path_test))
        
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
        cols = list(df_rawTrain.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        # Compute split sizes based on row count.
        # Example:
        #   if N = 100000 rows,
        #   num_train = 70000
        #   num_test  = 20000
        #   num_vali  = 10000
        
        
        #num_train = int(len(df_rawTrain) * self.train_split)
        #num_test = int(len(df_raw) * self.test_split)
        #num_vali = len(df_raw) - num_train - num_test
        
        num_train = int(len(df_rawTrain))
        num_test = int(len(df_rawTest))
        num_vali = int(len(df_rawEval))

        # As before, validation and test start positions are shifted left by seq_len
        # so that the first sample in each split can still access enough history.
        #border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        #border2s = [num_train, num_train + num_vali, len(df_raw)]
        #border1 = border1s[self.set_type]
        #border2 = border2s[self.set_type]

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
        #if self.features == 'M' or self.features == 'MS':
        #    cols_data = df_raw.columns[1:]
        #    df_data = df_raw[cols_data]
        #elif self.features == 'S':
        #    df_data = df_raw[[self.target]]
            
            
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_rawTrain.columns[1:]
            df_data = df_rawTrain[cols_data]
        elif self.features == 'S':
            df_data = df_rawTrain[[self.target]]

        # Fit scaler on train slice only.
        #if self.scale:
        #    train_data = df_data[border1s[0]:border2s[0]]
        #    self.scaler.fit(train_data.values)
        #    data = self.scaler.transform(df_data.values)
        #else:
        #    data = df_data.values
            
        if self.scale:
            train_data = df_rawTrain
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract time column for current split.
        #df_stamp = df_raw[[self.time_col_name]][border1:border2]
        #df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])

        if self.set_type == 0:
            df_raw = df_rawTrain
        elif self.set_type == 1:
            df_raw = df_rawEval
        else:
            df_raw = df_rawTest

        df_stamp = df_raw[[self.time_col_name]].copy()
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp[self.time_col_name].apply(lambda row: row.minute)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values

        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[self.time_col_name].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
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