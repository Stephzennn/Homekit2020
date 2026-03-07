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
df.show(5)

import pandas as pd

df = pd.read_parquet(
    "../data/processed/split_2020_02_10/train/date=2019-12-15"
)

bypeople = pd.read_parquet(
    "../data/processed/split_2020_02_10_by_user/train/date=2019-12-15"
)


print(df.head())
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