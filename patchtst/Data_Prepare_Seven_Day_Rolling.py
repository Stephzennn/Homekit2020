# here we will create 5 days worth of sample data , then we will extend to the whole dataset 
import pandas as pd 

import pyarrow.dataset as ds

#input_root = "./Homekit2020/data/processed/split_2020_02_10_by_user/test"
#output_csv = "./Homekit2020/data/processed/InfectionStatusAddedTrial.csv"

output_csvTrain = "./Homekit2020/data/processed/FullBypeople/WearableTrain.csv"

output_csvEval = "./Homekit2020/data/processed/FullBypeople/WearableEval.csv"

output_csvTest = "./Homekit2020/data/processed/FullBypeople/WearableTest.csv"

#output_csvReal = "./Homekit2020/data/processed/InfectionStatusAddedFullDataset.csv"

#fullDatasetFolder = "./Homekit2020/data/processed/full_dataset"

trainDataset = "./Homekit2020/data/processed/split_2020_02_10_by_user/train"

#EvalDataset = "./Homekit2020/data/processed/split_2020_02_10_by_user/eval" eval_7_day

EvalDataset = "./Homekit2020/data/processed/split_2020_02_10_by_user/eval_7_day_daily_features" 

EvalDatasetData = "../data/processed/split_2020_02_10_by_user/eval_7_day" 


TestDataset = "./Homekit2020/data/processed/split_2020_02_10_by_user/test"

labResultPath = "./Homekit2020/data/processed/lab_results_with_triggerdate.csv"

dataset = ds.dataset(
        EvalDataset,
        format="parquet",
        partitioning="hive"
    )
    
frag = next(dataset.get_fragments())
tbl = frag.to_table(batch_size=1)
row = tbl.slice(0, 1)

print(tbl.schema)

for col in [ "heart_rate", "missing_heart_rate", "missing_steps",
            "sleep_classic_0", "sleep_classic_1", "sleep_classic_2", "sleep_classic_3"]:
    arr = row[col][0].as_py()
    print(col, len(arr))
