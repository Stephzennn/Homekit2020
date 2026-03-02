import glob
import os

import click
import pandas as pd
import numpy as np

from src.data.utils import write_pandas_to_parquet, load_processed_table, read_parquet_to_pandas


@click.command()
@click.argument("out_path", type=click.Path(file_okay=False))
@click.argument("in_path", type=click.Path())
@click.option("--split_date", default=None)
@click.option("--end_date", default=None)
@click.option("--eval_frac", default=None)
@click.option(
    "--test_frac",
    default=0.5,
    help="Fraction of eval set that's reserved for testing",
)
@click.option("--activity_level", type=click.Choice(["day", "minute"]), default="minute")
@click.option("--separate_train_and_eval", is_flag=True)
def main(
    out_path,
    in_path,
    split_date=None,
    end_date=None,
    eval_frac=None,
    test_frac=0.5,
    activity_level="minute",
    separate_train_and_eval=False,
):
    """
    Splits a processed dataset into train/eval/test outputs.

    Two supported modes:
      - minute-level: reads parquet from `in_path` and uses `timestamp` as the time column
      - day-level: loads a named processed table and uses `date` as the time column

    Two supported splitting strategies:
      1) split_date-based: identify participants that have data on/after split_date, then hold out
         a fraction of those participants (test_frac) for testing on/after split_date.
      2) eval_frac-based: randomly select eval_frac of participants as the evaluation cohort, then
         carve out a test subset from within that cohort (test_frac * eval_frac).

    Output layout depends on `separate_train_and_eval`:
      - If True: writes train/, eval/, test/
      - If False: writes train_eval/, test/
    """

    # --- IMPORTANT JUNCTION: Load data + decide which time column is used ---
    if activity_level == "minute":
        df = read_parquet_to_pandas(in_path)
        timestamp_col = "timestamp"
    else:
        # Day-level loads a pre-defined processed table rather than the provided in_path
        df = load_processed_table("fitbit_day_level_activity")
        timestamp_col = "date"

    # --- IMPORTANT JUNCTION: Optional time truncation ---
    # Restrict data to strictly before end_date if provided
    if end_date:
        df = df[df[timestamp_col] < pd.to_datetime(end_date)]

    # --- IMPORTANT JUNCTION: Build the test mask (two different strategies) ---
    if split_date:
        # split_date strategy:
        #   - find participants with records on/after split_date
        #   - randomly select a fraction (test_frac) of those participants
        #   - test = those participants' records on/after split_date
        past_date_mask = df[timestamp_col] >= pd.to_datetime(split_date)
        participants_after_date = df[past_date_mask]["participant_id"].unique()
        np.random.shuffle(participants_after_date)

        test_participants = participants_after_date[
            : int(test_frac * len(participants_after_date))
        ]

        in_test_frac_mask = df["participant_id"].isin(test_participants) & past_date_mask

    elif eval_frac:
        # eval_frac strategy:
        #   - randomly shuffle all participants
        #   - first (test_frac * eval_frac) portion becomes test participants
        #   - next portion up to eval_frac becomes eval participants (only used if separate_train_and_eval=True)
        participant_ids = df["participant_id"].unique().values
        np.random.shuffle(participant_ids)

        test_index = int(test_frac * eval_frac * len(participant_ids))
        eval_index = int(eval_frac * len(participant_ids))

        test_participants = participant_ids[:test_index]
        eval_participants = participant_ids[test_index:eval_index]

        in_test_frac_mask = df["participant_id"].isin(test_participants)

    # NOTE: If neither split_date nor eval_frac is provided, in_test_frac_mask would be undefined.
    # (Leaving code behavior unchanged, as requested.)

    # --- IMPORTANT JUNCTION: Apply the test mask to split data ---
    train_eval = df[~in_test_frac_mask]
    test = df[in_test_frac_mask]

    # Build output paths (folder-like paths without extensions for parquet partitions)
    test_path = os.path.join(out_path, f"test_{activity_level}")

    to_write_dfs = [test]
    to_write_paths = [test_path]

    # --- IMPORTANT JUNCTION: Decide whether to split train vs eval separately ---
    if separate_train_and_eval:
        if split_date:
            # Time-based split inside the remaining train_eval set
            train = train_eval[train_eval[timestamp_col] < pd.to_datetime(split_date)]
            eval = train_eval[train_eval[timestamp_col] >= pd.to_datetime(split_date)]
        elif eval_frac:
            # Participant-based split using eval_participants
            eval_mask = train_eval["participant_id"].isin(eval_participants)
            train = train_eval[~eval_mask]
            eval = train_eval[eval_mask]

        train_path = os.path.join(out_path, f"train_{activity_level}")
        eval_path = os.path.join(out_path, f"eval_{activity_level}")

        to_write_dfs = to_write_dfs + [train, eval]
        to_write_paths = to_write_paths + [train_path, eval_path]
    else:
        # If not separating, keep a combined train_eval set
        train_eval_path = os.path.join(out_path, f"train_eval_{activity_level}")
        to_write_dfs += [train_eval]
        to_write_paths += [train_eval_path]

    # --- IMPORTANT JUNCTION: Ensure output directory exists ---
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # --- IMPORTANT JUNCTION: Normalize dtype for participant_id before writing ---
    df["participant_id"] = df["participant_id"].astype("string")

    # --- IMPORTANT JUNCTION: Write outputs (parquet for minute-level, CSV for day-level) ---
    if activity_level == "minute":
        # Writes partitioned parquet by date (efficient loading / filtering by day)
        for df, path in zip(to_write_dfs, to_write_paths):
            write_pandas_to_parquet(
                df,
                path,
                partition_cols=["date"],
                overwrite=True,
                engine="fastparquet",
            )
    else:
        # Day-level outputs are written as CSV
        for df, path in zip(to_write_dfs, to_write_paths):
            df.to_csv(path + ".csv", index=False)


if __name__ == "__main__":
    main()