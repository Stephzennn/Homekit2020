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

EvalDataset = "./Homekit2020/data/processed/split_2020_02_10_by_user/eval"

TestDataset = "./Homekit2020/data/processed/split_2020_02_10_by_user/test"

labResultPath = "./Homekit2020/data/processed/lab_results_with_triggerdate.csv"

"""
dataset = ds.dataset(
    fullDatasetFolder,
    format="parquet",
    partitioning="hive"
)

table = dataset.to_table(
    filter=(
        (ds.field("date") >= "2020-02-05") &
        (ds.field("date") <= "2020-02-05")
    )
)

bypeople = table.to_pandas()
"""

def Prepare_Data_for_PatchTST(bypeople):
    
    bypeople["InfectionStatus"] = 0
    participantIDD = set(bypeople["participant_id"])

    pp = 2
    for x in participantIDD:
        if pp >= 1:
            print("Started")
        participantId = x
        labResult = pd.read_csv(labResultPath) 

        #labResult.head(20)

        dummy = labResult.loc[
            (labResult["participant_id"] == participantId) &
            (labResult["result"] == "Detected")
        ].copy()

        # REMOVE THIS LINE LATER ON 
        #if dummy.shape[0] == 1:
        #    dummy = pd.concat([dummy, dummy.iloc[[-1]]], ignore_index=True)

        #dummy

        dummy["trigger_datetime"] = pd.to_datetime(dummy["trigger_datetime"]).dt.floor("min")

        # edge case, more than one infection.
        dummyTriggerDates = dummy["trigger_datetime"]
        #dummyTriggerDates



        # This updates the infection column of our dataset
        for dates in dummyTriggerDates:
            #print(dates)
            find = bypeople.loc[
            (bypeople["participant_id"] == participantId) &
            (bypeople["timestamp"] == dates) 
            ]
            if (find["InfectionStatus"] == 0).any():
                bypeople.loc[
                    (bypeople["participant_id"] == participantId) &
                    (bypeople["timestamp"] == dates),
                    "InfectionStatus"
                ] = 1
        pp = pp - 1
    #bypeople.to_csv(output_csv, index=False) 
    #bypeople.to_csv(output_csvReal, index=False)
    print("Done")
        


# Parallelize the function to improve GPU utilization and speed up execution.
def Prepare_Data_for_PatchTSTV2(bypeople, OutputPath):
    print("Inside")
    bypeople = bypeople.copy()
    bypeople["InfectionStatus"] = 0
    bypeople["timestamp"] = pd.to_datetime(bypeople["timestamp"]).dt.floor("min")

    labResult = pd.read_csv(labResultPath)

    dummy = labResult.loc[
        labResult["result"] == "Detected",
        ["participant_id", "trigger_datetime"]
    ].copy()

    dummy["trigger_datetime"] = pd.to_datetime(dummy["trigger_datetime"]).dt.floor("min")
    dummy = dummy.rename(columns={"trigger_datetime": "timestamp"})
    dummy = dummy.drop_duplicates(subset=["participant_id", "timestamp"])

    match_idx = bypeople.set_index(["participant_id", "timestamp"]).index.isin(
        dummy.set_index(["participant_id", "timestamp"]).index
    )

    bypeople.loc[match_idx, "InfectionStatus"] = 1

    # Here the column is [participant_id,timestamp,steps,heart_rate,missing_heartrate,missing_steps,sleep_classic_0,sleep_classic_1,sleep_classic_2,sleep_classic_3,date,InfectionStatus]
    bypeople.drop(columns=["date", "participant_id", "__index_level_0__"], inplace=True)
    bypeople.rename(columns={"timestamp": "date"}, inplace=True)
    
    # Here the final column should be [timestamp steps heart_rate missing_heartrate missing_steps sleep_classic_0 sleep_classic_1 sleep_classic_2 sleep_classic_3 InfectionStatus]
    #bypeople.to_csv(output_csvReal, index=False) output_csv
    #bypeople.to_csv(output_csv, index=False) 
    
    bypeople.to_csv(OutputPath, index=False) 
    print("Done")

def main():
    print(1)
    dataset = ds.dataset(
        trainDataset,
        format="parquet",
        partitioning="hive"
    )
    print(2)
    # Trial
    """
    table = dataset.to_table(
        filter=(
            (ds.field("date") >= "2020-02-05") &
            (ds.field("date") <= "2020-02-05")
        )
    )
    
    
    """
    #Testing
    #table = dataset.to_table(
    #    filter=(
    #        (ds.field("date") >= "2019-12-15") &
    #        (ds.field("date") <= "2019-12-15")
    #    )
    #)
    table = dataset.to_table()
    """
    #Final
    table = dataset.to_table(
        filter=(
            (ds.field("date") >= "2019-12-15") &
            (ds.field("date") <= "2020-06-31")
        )
    )
    """
    print(3)
    bypeople = table.to_pandas()
    print(4)
    Prepare_Data_for_PatchTSTV2(bypeople, output_csvTrain)

if __name__ == "__main__":
    main()