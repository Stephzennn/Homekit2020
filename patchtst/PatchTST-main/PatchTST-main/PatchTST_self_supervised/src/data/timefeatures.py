from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# This file defines a small framework for converting timestamps into numeric "time features".
#
# Why this exists:
# ----------------
# Time-series models often benefit from explicit calendar information, such as:
# - hour of day
# - day of week
# - month of year
# - minute of hour
#
# Instead of feeding raw datetime strings into the model, this code converts a
# pandas DatetimeIndex into normalized numeric arrays.
#
# The normalization is usually to the range approximately [-0.5, 0.5].
# That is helpful because:
# - features stay numerically small
# - different calendar units are on roughly similar scales
# - many neural models train more stably with normalized inputs
#
# Example:
# --------
# If you have timestamps like:
#   2020-01-01 00:00:00
#   2020-01-01 01:00:00
#   2020-01-01 02:00:00
#
# then time_features(...) might return rows for:
# - hour of day
# - day of week
# - day of month
# - day of year
#
# producing an array of shape:
#   [num_time_features, num_timestamps]
#
# Important input requirement:
# ----------------------------
# All of these functions assume the input is something equivalent to:
#   pd.DatetimeIndex
#
# or something that can be interpreted by pandas as one.
#
# So upstream code usually does:
#   pd.to_datetime(...)
#
# before calling these feature extractors.
#
# Important output-shape note:
# ----------------------------
# The final helper function:
#
#   time_features(dates, freq='h')
#
# returns:
#   np.ndarray with shape [num_time_features, len(dates)]
#
# Many dataset classes then transpose that to:
#   [len(dates), num_time_features]
#
# so that each row corresponds to one timestamp.
#
# This file does NOT read CSVs directly.
# It operates only on already-parsed datetime values.


class TimeFeature:
    # Base class for all time feature encoders.
    #
    # Conceptually:
    #   a TimeFeature is a callable object that maps a DatetimeIndex
    #   to one numeric feature array.
    #
    # Example subclass:
    #   HourOfDay(index) -> normalized hour values
    #
    # The base class itself does not implement any real feature logic.
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # This method is meant to be overridden in subclasses.
        #
        # Input:
        #   index: pandas DatetimeIndex, shape conceptually [N timestamps]
        #
        # Output:
        #   numpy array of shape [N]
        #
        # Since this base class is abstract-like, calling it directly
        # would do nothing useful.
        pass

    def __repr__(self):
        # This returns a readable string representation like:
        #   HourOfDay()
        #   DayOfWeek()
        #
        # Useful for debugging and printing lists of feature objects.
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract the second component from each timestamp.
        #
        # index.second gives integers in [0, 59].
        #
        # Dividing by 59.0 maps that to [0, 1].
        # Subtracting 0.5 maps that to [-0.5, 0.5].
        #
        # Example:
        #   second = 0   -> -0.5
        #   second = 59  ->  0.5
        #
        # Output shape:
        #   [len(index)]
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract minute from each timestamp.
        #
        # index.minute is in [0, 59].
        # Normalize to approximately [-0.5, 0.5].
        #
        # Example:
        #   minute = 0   -> -0.5
        #   minute = 30  -> ~0.0085
        #   minute = 59  -> 0.5
        #
        # Output shape:
        #   [len(index)]
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract hour from each timestamp.
        #
        # index.hour is in [0, 23].
        # Normalize to [-0.5, 0.5].
        #
        # Example:
        #   hour = 0   -> -0.5
        #   hour = 12  -> ~0.0217
        #   hour = 23  -> 0.5
        #
        # Output shape:
        #   [len(index)]
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract day of week from each timestamp.
        #
        # pandas uses:
        #   Monday = 0
        #   Sunday = 6
        #
        # Normalization:
        #   divide by 6.0 -> [0, 1]
        #   subtract 0.5  -> [-0.5, 0.5]
        #
        # Output shape:
        #   [len(index)]
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract day-of-month from each timestamp.
        #
        # index.day is in [1, 31].
        # Subtract 1 first so the range becomes [0, 30].
        # Divide by 30.0 -> [0, 1]
        # Subtract 0.5   -> [-0.5, 0.5]
        #
        # Note:
        # This assumes 31-day-style normalization.
        # For months shorter than 31 days, the effective max will be below 0.5.
        #
        # Output shape:
        #   [len(index)]
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract day-of-year from each timestamp.
        #
        # index.dayofyear is in [1, 365] or [1, 366] for leap years.
        # This code normalizes using 365.0, so leap years are slightly compressed.
        #
        # Steps:
        #   subtract 1   -> [0, 364]
        #   divide 365.0 -> approx [0, 1)
        #   subtract 0.5 -> approx [-0.5, 0.5)
        #
        # Output shape:
        #   [len(index)]
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract month from each timestamp.
        #
        # index.month is in [1, 12].
        # Subtract 1 -> [0, 11]
        # Divide by 11.0 -> [0, 1]
        # Subtract 0.5 -> [-0.5, 0.5]
        #
        # Output shape:
        #   [len(index)]
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract ISO calendar week number from each timestamp.
        #
        # index.isocalendar().week usually gives values in [1, 52] or sometimes 53.
        # This code normalizes assuming 52 weeks:
        #   subtract 1 -> [0, 51]
        #   divide by 52.0
        #   subtract 0.5
        #
        # Important note:
        # ISO calendars can include week 53, so values may slightly exceed
        # the intended normalized range in those cases.
        #
        # Output shape:
        #   [len(index)]
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    # This dictionary maps pandas offset types to the set of time features
    # that make sense for that resolution.
    #
    # Example intuition:
    # - If your data are yearly, there is no meaningful hour/day/minute information.
    # - If your data are hourly, hour-of-day and day-based features make sense.
    # - If your data are minutely, minute-of-hour and larger-scale calendar features make sense.
    #
    # Each value is a LIST OF CLASSES, not instances.
    # The function later instantiates them with cls().
    features_by_offsets = {
        # Yearly data:
        #   no additional calendar decomposition is used here.
        offsets.YearEnd: [],

        # Quarterly data:
        #   month-of-year can still be meaningful.
        offsets.QuarterEnd: [MonthOfYear],

        # Monthly data:
        #   month-of-year is a natural seasonal feature.
        offsets.MonthEnd: [MonthOfYear],

        # Weekly data:
        #   day-of-month and week-of-year may still be informative.
        offsets.Week: [DayOfMonth, WeekOfYear],

        # Daily data:
        #   weekday, day-of-month, day-of-year are relevant.
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],

        # Business-day data:
        #   similar to daily, but skipping weekends in the source series.
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],

        # Hourly data:
        #   add hour-of-day on top of day-based seasonality.
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],

        # Minute-level data:
        #   add minute-of-hour, then broader cycles.
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],

        # Second-level data:
        #   add second-of-minute, then minute/hour/day cycles.
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    # Convert a frequency string like:
    #   'h', 'H', '15min', 'T', 'D', 'W', 'M'
    # into a pandas offset object.
    #
    # Examples:
    #   to_offset('h')      -> Hour
    #   to_offset('15min')  -> Minute(15)
    #   to_offset('D')      -> Day
    offset = to_offset(freq_str)

    # Find the first offset type whose class matches the parsed offset.
    #
    # Example:
    #   if freq_str='15min', offset is a Minute-type offset,
    #   so isinstance(offset, offsets.Minute) is True
    #
    # Then instantiate each listed TimeFeature class and return them.
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # If the frequency is unsupported, raise a readable error explaining valid choices.
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    # This is the main convenience function used by the dataset loaders.
    #
    # Input:
    #   dates:
    #       usually a pandas DatetimeIndex or something convertible to one
    #       shape conceptually: [N timestamps]
    #
    #   freq:
    #       frequency string such as:
    #           'h'      -> hourly
    #           '15min'  -> 15-minute data
    #           'D'      -> daily
    #
    # Process:
    #   1. Determine which time feature extractors are appropriate for this frequency
    #   2. Apply each extractor to the full dates array
    #   3. Stack them vertically with np.vstack
    #
    # Output:
    #   numpy array of shape:
    #       [num_time_features, len(dates)]
    #
    # Example for hourly data:
    #   if dates has length N and freq='h',
    #   the selected features are:
    #       HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
    #
    #   so output shape is:
    #       [4, N]
    #
    # Many dataset loaders then transpose this to:
    #       [N, 4]
    #
    # so that each row aligns with one timestamp/sample.
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


