from typing import Optional
import datetime as dt
import gc
import glob
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from src.data.utils import process_minute_level_pandas
from src.utils import get_logger

logger = get_logger(__name__)


def explode_str_column(
    df: pd.DataFrame,
    target_col: str,
    freq: str = "min",
    dur: str = "1D",
    sep_char: str = " ",
    date_col: str = "dt",
    dtype: str = "Int32",
    participant_col: str = "id_participant_external",
    rename_participant_id_column: str = "participant_id",
    rename_target_column: Optional[str] = None,
    start_col: Optional[str] = None,
    dur_col: Optional[str] = None,
    clip_max: int = 200,
    single_val: bool = False,
) -> pd.DataFrame:
    """
    Expands a column that encodes minute-level values as a string into a proper minute-level
    time-indexed DataFrame.

    Typical use-case: a record contains a start time + a string like "0 0 1 2 ..." representing
    per-minute values (sleep class, steps, heart rate, etc.). This function:
      1) builds the minute-level timestamp index
      2) splits the string into per-minute values
      3) explodes into long format: one row per minute
      4) returns a timestamp-indexed series (DataFrame) of that variable
    """

    # Decide output column name and participant id column name (if renamed)
    val_col_name = rename_target_column if rename_target_column else target_col
    pid_col_name = (
        rename_participant_id_column if rename_participant_id_column else participant_col
    )

    # If there is no data for this user, return an empty time-indexed frame
    if df.empty:
        return (
            pd.DataFrame(
                columns=["timestamp", val_col_name],
                index=pd.DatetimeIndex([]),
            )
            .set_index("timestamp")
        )

    # --- IMPORTANT JUNCTION: Build per-row minute-level timestamp arrays ---
    # For each record, get_new_index generates the full minute-by-minute timestamps to align with
    # the values inside target_col (minute_level_str).
    df["timestamp"] = df.apply(
        get_new_index,
        target_column=target_col,
        start_col=start_col,
        dur_col=dur_col,
        freq=freq,
        dur=dur,
        axis=1,
        date_col=date_col,
    )

    # --- IMPORTANT JUNCTION: Convert the encoded string values to a list of per-minute values ---
    # If single_val=False: split the string into a list using sep_char.
    # If single_val=True: repeat the single value to match the length of the timestamp array.
    if not single_val:
        df["val"] = df[target_col].str.split(sep_char)
    else:
        df["val"] = df.apply(lambda x: [x[target_col]] * len(x["timestamp"]), axis=1)

    # --- IMPORTANT JUNCTION: Explode into long format (one row per minute) ---
    df = df[["timestamp", "val"]].explode(["timestamp", "val"])

    # Convert to numeric, downcast to save memory, and clip extreme values
    df["val"] = pd.to_numeric(df["val"], downcast="unsigned").clip(upper=clip_max)

    # Sort by time, then deduplicate timestamps (keep last) to avoid overlapping/conflicts
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="last")

    # Rename the generic 'val' column to the requested output name (heart_rate, steps, sleep_classic, etc.)
    df = df.rename(columns={"val": val_col_name})

    # Return as a timestamp-indexed frame
    df = df.set_index("timestamp").sort_index()
    return df


def get_new_index(
    item: dict,
    target_column: str,
    freq: str = "min",
    dur: str = "1D",
    sep_char: str = " ",
    date_col: str = "dt",
    start_col: Optional[str] = None,
    dur_col: Optional[str] = None,
) -> list:
    """
    Generates a per-minute timestamp index for one record.

    Two modes:
      - If start_col is provided: start = item[start_col], duration = item[dur_col] minutes
      - Else: start = item[date_col], duration = dur (e.g., "1D")

    Returns: numpy array / list-like of timestamps corresponding to each minute's value.
    """

    # --- IMPORTANT JUNCTION: Determine start/end timestamps ---
    if start_col:
        start = item[start_col]
        start_ts = pd.to_datetime(start).round(freq)
        end_ts = start_ts + pd.to_timedelta(item[dur_col], unit=freq)
    else:
        start_ts = pd.to_datetime(item[date_col])
        end_ts = start_ts + pd.to_timedelta(dur)

    # Create minute-level range; closed="left" means end_ts is excluded (aligns with "duration" semantics)
    new_index = pd.date_range(start_ts, end_ts, freq=freq, closed="left").values
    return new_index


CHUNKSIZE = "1GB"
PARTITION_SIZE = "1GB"


def read_raw_pandas(path, set_dtypes=None):
    """
    Reads a parquet file into pandas, applies basic dtype cleanup, and indexes by participant id.
    """
    logger.info("Reading...")
    df = pd.read_parquet(path, engine="pyarrow")

    # --- IMPORTANT JUNCTION: Category-encode participant IDs to save memory ---
    df["id_participant_external"] = df["id_participant_external"].astype("category")

    # Optional: enforce specific dtypes for columns (memory/performance control)
    if set_dtypes:
        for k, v in set_dtypes.items():
            df[k] = df[k].astype(v)

    # Drop missing rows and index by participant id for efficient per-user slicing
    return df.dropna().set_index("id_participant_external")


def safe_loc(df, ind):
    """
    Safe wrapper for df.loc[[ind]]: returns empty DataFrame (same columns) if ind is missing.
    """
    try:
        return df.loc[[ind]]
    except KeyError:
        return pd.DataFrame(columns=df.columns)


COLUMNS = [
    "date",
    "timestamp",
    "heart_rate",
    "steps",
    "missing_heart_rate",
    "missing_steps",
    "sleep_classic_0",
    "sleep_classic_1",
    "sleep_classic_2",
    "sleep_classic_3",
]


@click.command()
@click.argument("sleep_in_path", type=click.Path(exists=True))
@click.argument("steps_in_path", type=click.Path(exists=True))
@click.argument("heart_rate_in_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def main(
    sleep_in_path: str,
    steps_in_path: str,
    heart_rate_in_path: str,
    out_path: str,
) -> None:
    """
    Pipeline entrypoint:
      - Load raw sleep / steps / heart rate parquet
      - For each user with steps:
          * explode minute-level strings into timestamp-indexed series
          * join into a unified minute-level table
          * run process_minute_level_pandas (feature engineering / missingness flags / date extraction)
      - Concatenate all users and write partitioned parquet by date
    """

    start = time.time()

    logger.info("Loading sleep...")
    sleep = read_raw_pandas(sleep_in_path)

    logger.info("Loading heart rate...")
    hr = read_raw_pandas(heart_rate_in_path)

    logger.info("Loading steps...")
    steps = read_raw_pandas(steps_in_path)

    # --- IMPORTANT JUNCTION: Choose cohort = users who have steps data ---
    users_with_steps = steps.index.unique()

    logger.info("Processing users...")
    all_results = []

    # --- IMPORTANT JUNCTION: Per-user processing loop ---
    for user in tqdm(users_with_steps.values):
        # Explode sleep minute-level string using explicit start + duration columns
        exploded_sleep = explode_str_column(
            safe_loc(sleep, user),
            target_col="minute_level_str",
            rename_target_column="sleep_classic",
            start_col="main_start_time",
            dur_col="main_in_bed_minutes",
            dtype=pd.Int8Dtype(),
        )

        # Explode heart rate minute-level string using dt + 1-day default window
        exploded_hr = explode_str_column(
            safe_loc(hr, user),
            target_col="minute_level_str",
            rename_target_column="heart_rate",
            dtype=pd.Int8Dtype(),
        )

        # Explode steps minute-level string using dt + 1-day default window
        exploded_steps = explode_str_column(
            safe_loc(steps, user),
            target_col="minute_level_str",
            rename_target_column="steps",
            dtype=pd.Int8Dtype(),
        )

        # --- IMPORTANT JUNCTION: Merge modalities on timestamp ---
        steps_and_hr = exploded_steps.join(exploded_hr, how="left")
        merged = steps_and_hr.join(exploded_sleep, how="left")

        # --- IMPORTANT JUNCTION: Central processing/feature-engineering step ---
        # process_minute_level_pandas likely:
        #   - creates date column
        #   - adds missingness flags
        #   - expands sleep_classic into one-hot columns sleep_classic_0..3
        #   - ensures consistent index/time coverage
        processed = process_minute_level_pandas(minute_level_df=merged)

        # Keep datatypes in check (memory + downstream consistency)
        processed["heart_rate"] = processed["heart_rate"].astype(pd.Int16Dtype())

        # Attach participant ID for later grouping/analysis
        processed["participant_id"] = user
        all_results.append(processed)

    # Concatenate all participants into one big table
    all_results = pd.concat(all_results)

    # --- IMPORTANT JUNCTION: Ensure one-hot sleep columns have no missing values ---
    # (sleep may be missing for some timestamps/users; fill as False for boolean indicators)
    all_results["sleep_classic_0"] = all_results["sleep_classic_0"].fillna(False)
    all_results["sleep_classic_1"] = all_results["sleep_classic_1"].fillna(False)
    all_results["sleep_classic_2"] = all_results["sleep_classic_2"].fillna(False)
    all_results["sleep_classic_3"] = all_results["sleep_classic_3"].fillna(False)

    # --- IMPORTANT JUNCTION: Write output partitioned by date for efficient querying ---
    all_results.to_parquet(path=out_path, partition_cols=["date"], engine="fastparquet")

    end = time.time()
    print("Time elapsed", end - start)


if __name__ == "__main__":
    main()