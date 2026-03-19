


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


def get_dls(params):
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
        root_path = './Homekit2020/data/processed/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
            datasetCls=Dataset_HomeKitWearableV3,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': ['WearableTrain.csv', 'WearableEval.csv', 'WearableTest.csv'],
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )
    """
    elif params.dset == 'wearable':

    root_path = './Homekit2020/data/processed/'
    size = [params.context_points, 0, params.target_points]

    dls = DataLoaders(
        datasetCls=Dataset_Custom,
        dataset_kwargs={
            'root_path': root_path,
            'data_path': 'wearable_train.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        },

        valid_dataset_kwargs={
            'root_path': root_path,
            'data_path': 'wearable_eval.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        },

        test_dataset_kwargs={
            'root_path': root_path,
            'data_path': 'wearable_test.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        },

        batch_size=params.batch_size,
        workers=params.num_workers,
    )
    """

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
    dls.vars = dls.train.dataset[0][0].shape[1]

    # Save the context window length directly from params
    dls.len = params.context_points

    # Save target dimensionality
    dls.c = dls.train.dataset[0][1].shape[0]

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
        dset = 'Wearable'          # dataset name
        context_points = 720       # input sequence length
        target_points = 48         # forecast horizon
        batch_size = 32            # batch size
        num_workers = 8            # dataloader workers
        with_ray = False           # appears unused here, likely for other workflows
        features = 'M'             # multivariate mode

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



"""


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
    # dataset is assume to have dimension len x nvars
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

"""