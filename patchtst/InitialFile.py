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

len(datasurvey.columns)

fitbitDay.head()

for item in Path.cwd().iterdir():
    size = item.stat().st_size
    print(f"{item.name}  |  {size} bytes")