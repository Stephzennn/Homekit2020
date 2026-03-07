
#import pandas as pd

"""
bypeopleTest = pd.read_parquet(
    "./Homekit2020/data/processed/split_2020_02_10_by_user/test"
)



print("start")
bypeopleTest = pd.read_parquet(
    "./Homekit2020/data/processed/split_2020_02_10_by_user/test",
    engine="pyarrow"
)

print("finish")
bypeopleTest = bypeopleTest.sort_values("date")

bypeopleTest.to_csv(
    "/Homekit2020/data/processed/by_people_test_all_dates.csv",
    index=False
)
#/home/hice1/ezg6/projects/Homekit2020/data/processed

"""

#===

#===

import os
import glob
import pandas as pd
"""
input_root = "./Homekit2020/data/processed/split_2020_02_10_by_user/test"
output_csv = "./Homekit2020/data/processed/by_people_test_all_dates.csv"

date_dirs = sorted(glob.glob(os.path.join(input_root, "date=*")))

first = True

for d in date_dirs:
    print(f"Reading {d} ...")
    df = pd.read_parquet(d, engine="pyarrow")

    # Optional: sort if needed
    if "date" in df.columns:
        df = df.sort_values("date")

    df.to_csv(output_csv, mode="w" if first else "a", header=first, index=False)
    first = False

    print(f"Wrote {len(df)} rows from {d}")
    del df

print("Done.")
"""



#For the training data split by users

input_root = "./Homekit2020/data/processed/split_2020_02_10_by_user/train"
output_csv = "./Homekit2020/data/processed/by_people_train_all_dates.csv"

date_dirs = sorted(glob.glob(os.path.join(input_root, "date=*")))

first = True

for d in date_dirs:
    print(f"Reading {d} ...")
    df = pd.read_parquet(d, engine="pyarrow")

    # Optional: sort if needed
    if "date" in df.columns:
        df = df.sort_values("date")

    df.to_csv(output_csv, mode="w" if first else "a", header=first, index=False)
    first = False

    print(f"Wrote {len(df)} rows from {d}")
    del df

print("Done.")