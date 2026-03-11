from pathlib import Path
 
import os

print("Current working directory:", Path.cwd())


#print(Path.home().parents[2]/ "scratch" / "ezg6" / "homekit2020")
DATA_PATH = Path.home()/"scratch"/ "homekit2020-1.0" # / "homekit2020"  #.parents[2]#


os.chdir(DATA_PATH)
print("Current working directory:", Path.cwd())

daily_survey_onehot_data_path = DATA_PATH / "daily_surveys_onehot.csv"

fitbit_day_level_activity_path = DATA_PATH / 'fitbit_day_level_activity.csv'

print(daily_survey_onehot_data_path)

import pandas as pd

datasurvey = pd.read_csv(daily_survey_onehot_data_path)

fitbitDay = pd.read_csv(fitbit_day_level_activity_path )

hourlyTrainDataPath = DATA_PATH /data/processed/split_2020_02_10/train_7_day_hourly

len(datasurvey.columns)

fitbitDay.head()

for item in Path.cwd().iterdir():
    size = item.stat().st_size
    print(f"{item.name}  |  {size} bytes")
    
    
    
    
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("../data/processed/split_2020_02_10/train_7_day_hourly")

df.printSchema()
df.columns

import pandas as pd

df = pd.read_parquet(
   
    "../data/processed/split_2020_02_10_by_user/train_7_day/part-00000-e13c451f-9bc0-4dda-84a7-b9971e0a7924-c000.snappy.parquet"
)

dfCSV = pd.read_csv(
    "../data/processed/fitbit_day_level_activity.csv"
)


print(dfCSV.columns)


dfCSV2 = pd.read_csv(
    "../data/processed/fitbit_day_level_activity.csv"  
)
print(dfCSV2.head())

dfCSV2.shape

small3 = dfCSV2.head(30)


dfCSV3 = pd.read_csv(
    "../data/processed/daily_surveys_onehot.csv"
)
print(dfCSV3.head())

small = dfCSV3.head()

bypeople = pd.read_parquet(
    "../data/processed/full_dataset/date=2019-12-15"
)
#ff1f623ad0ad4d13816426b7fa6229c4.parquet

bypeople =  pd.read_parquet(
    "../data/processed/split_2020_02_10_by_user/train_7_day/.part-00000-e13c451f-9bc0-4dda-84a7-b9971e0a7924-c000.snappy.parquet"
)



#/home/hice1/ezg6/projects/Homekit2020/data/processed/split_2020_02_10_by_user/train_7_day/.part-00000-e13c451f-9bc0-4dda-84a7-b9971e0a7924-c000.snappy.parquet

#import pandas as pd

df = pd.read_parquet(
    "part-00000-e13c451f-9bc0-4dda-84a7-b9971e0a7924-c000.snappy.parquet"
)

print(bypeople.head())

dd = bypeople[bypeople["participant_id"].str.contains("18ebe4d026fc2782df611fbfca8e11ff")]

print(df.head(10))
print(df.columns)


participants = set(bypeople['participant_id'])

len(participants) 


print(bypeople.head())

print(bypeople.tail())


import pandas as pd

df = pd.read_csv(
    "../data/processed/by_people_train_all_dates.csv",
    nrows=10
)

print(df)