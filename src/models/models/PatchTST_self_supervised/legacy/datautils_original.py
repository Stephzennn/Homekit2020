# =============================================================================
# LEGACY — datautils old versions
# Kept for reference only. Do NOT import or run directly.
#
# Contains three previous implementations that were replaced by the active
# datautils.py:
#
#   Block 1: Old Wearable loaders using Dataset_HomeKitWearableV3 and
#            Dataset_HomeKitParquetClassification (CSV / parquet-backed).
#
#   Block 2: Another old Wearable loader using Dataset_Custom with
#            separate CSV files per split.
#
#   Block 3: The original datautils.py — ETT/standard benchmarks only,
#            no Wearable support, no DataLoadersV2, no DictDataset.
#            Uses the simple dls.vars / dls.len / dls.c extraction pattern.
# =============================================================================


# ---------------------------------------------------------------------------
# Block 1 — Old Wearable loaders (HomeKitWearableV3 / HomeKitParquetClassification)
# ---------------------------------------------------------------------------

#    elif params.dset == 'Wearable':
#        root_path = './Homekit2020/data/processed/'
#        size = [params.context_points, 0, params.target_points]
#
#        dls = DataLoaders(
#            datasetCls=Dataset_HomeKitWearableV3,
#            dataset_kwargs={
#                'root_path': root_path,
#                'data_path': ['WearableTrain.csv', 'WearableEval.csv', 'WearableTest.csv'],
#                'features': params.features,
#                'scale': True,
#                'size': size,
#                'use_time_features': params.use_time_features
#            },
#            batch_size=params.batch_size,
#            workers=params.num_workers,
#        )
#    elif params.dset == 'Wearable':
#        root_path = './Homekit2020/data/processed/split_2020_02_10_by_user/'
#        dls = DataLoaders(
#            datasetCls=Dataset_HomeKitParquetClassification,
#            dataset_kwargs={
#                'root_path': root_path,
#                'data_path': 'train_7_day',
#                'split': 'train',
#                'scale': False,
#                'window_onset_min': 0,
#                'window_onset_max': 0,
#            },
#            valid_dataset_kwargs={
#                'root_path': root_path,
#                'data_path': 'eval_7_day',
#                'split': 'val',
#                'scale': False,
#                'window_onset_min': 0,
#                'window_onset_max': 0,
#            },
#            test_dataset_kwargs={
#                'root_path': root_path,
#                'data_path': 'test_7_day',
#                'split': 'test',
#                'scale': False,
#                'window_onset_min': 0,
#                'window_onset_max': 0,
#            },
#            batch_size=params.batch_size,
#            workers=params.num_workers,
#        )


# ---------------------------------------------------------------------------
# Block 2 — Old Wearable loader using Dataset_Custom with per-split CSVs
# ---------------------------------------------------------------------------

#    elif params.dset == 'Wearable':
#        root_path = './Homekit2020/data/processed/'
#        size = [params.context_points, 0, params.target_points]
#
#        dls = DataLoaders(
#            datasetCls=Dataset_Custom,
#            dataset_kwargs={
#                'root_path': root_path,
#                'data_path': 'wearable_train.csv',
#                'features': params.features,
#                'scale': True,
#                'size': size,
#                'use_time_features': params.use_time_features
#            },
#            valid_dataset_kwargs={
#                'root_path': root_path,
#                'data_path': 'wearable_eval.csv',
#                'features': params.features,
#                'scale': True,
#                'size': size,
#                'use_time_features': params.use_time_features
#            },
#            test_dataset_kwargs={
#                'root_path': root_path,
#                'data_path': 'wearable_test.csv',
#                'features': params.features,
#                'scale': True,
#                'size': size,
#                'use_time_features': params.use_time_features
#            },
#            batch_size=params.batch_size,
#            workers=params.num_workers,
#        )


# ---------------------------------------------------------------------------
# Block 3 — Original datautils.py (ETT benchmarks only, no Wearable support)
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *


DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange'
        ]

def get_dls(params):

    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

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

    # dataset is assumed to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls


if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
