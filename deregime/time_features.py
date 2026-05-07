from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


# ========================== TIME FEATURES ========================================================
class TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    def __call__(self, idx):
        return idx.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    def __call__(self, idx):
        return idx.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    def __call__(self, idx):
        return idx.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    def __call__(self, idx):
        return idx.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    def __call__(self, idx):
        return (idx.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    def __call__(self, idx):
        return (idx.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    def __call__(self, idx):
        return (idx.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    def __call__(self, idx):
        # robust to pandas versions
        iso = idx.isocalendar()
        week = iso["week"].to_numpy() if hasattr(iso, "__getitem__") else idx.week
        return (week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }
    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]
    raise RuntimeError(f"Unsupported frequency {freq_str}.")


def time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


def decompose_time(time: np.ndarray, freq: str) -> np.ndarray:
    df_stamp = pd.DataFrame(pd.to_datetime(time), columns=["date"])
    freq_scores = {"m": 0, "w": 1, "b": 2, "d": 2, "h": 3, "t": 4, "s": 5}
    max_score = max(freq_scores.values())
    df_stamp["month"] = df_stamp.date.dt.month
    if freq_scores.get(freq.lower(), max_score) >= 1:
        df_stamp["day"] = df_stamp.date.dt.day
    if freq_scores.get(freq.lower(), max_score) >= 2:
        df_stamp["weekday"] = df_stamp.date.dt.weekday
    if freq_scores.get(freq.lower(), max_score) >= 3:
        df_stamp["hour"] = df_stamp.date.dt.hour
    if freq_scores.get(freq.lower(), max_score) >= 4:
        df_stamp["minute"] = df_stamp.date.dt.minute
    if freq_scores.get(freq.lower(), max_score) >= 5:
        df_stamp["second"] = df_stamp.date.dt.second
    return df_stamp.drop(["date"], axis=1).values


# ---------- NEW: cyclical sin/cos encodings ----------
def _sincos(x: np.ndarray, period: int) -> np.ndarray:
    s = np.sin(2 * np.pi * x / period)
    c = np.cos(2 * np.pi * x / period)
    return np.stack([s, c], axis=-1)  # (..., 2)


def cyclical_time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    """
    Returns features stacked as:
    [ sin/cos(minute?), sin/cos(hour), sin/cos(weekday), sin/cos(month), sin/cos(dayofyear) ]
    Only includes minute if the data is sub-hourly.
    """
    minute = dates.minute.values
    hour = dates.hour.values
    dow = dates.weekday.values
    month = dates.month.values
    doy = dates.dayofyear.values

    feats = []
    # minute only if frequency is minute-or-finer
    if to_offset(freq).nanos <= to_offset("T").nanos:
        feats.append(_sincos(minute, 60))  # (T,2)
    feats.append(_sincos(hour, 24))
    feats.append(_sincos(dow, 7))
    feats.append(_sincos(month - 1, 12))  # 1..12 -> 0..11
    feats.append(_sincos(doy - 1, 365))  # 1..365 -> 0..364

    F = np.concatenate(feats, axis=-1).astype(np.float32)  # (T, C)
    return F


def get_time_mark(time_stamp: np.ndarray, timeenc: int, freq: str) -> np.ndarray:
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq).reshape(
            origin_size + (-1,)
        )
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq).T
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 2:
        origin_size = time_stamp.shape
        data_stamp = np.arange(np.prod(time_stamp.shape)).reshape(origin_size)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 3:
        dt = pd.to_datetime(time_stamp.flatten())
        cyc = cyclical_time_features(dt, freq=freq)  # (T, C_cyc)
        data_stamp = cyc.reshape(origin_size + (-1,))
    elif timeenc == 4:
        # hybrid: concat legacy(1) and cyc(3)
        dt = pd.to_datetime(time_stamp.flatten())
        legacy = time_features(dt, freq=freq).T  # (T, C_legacy)
        cyc = cyclical_time_features(dt, freq=freq)  # (T, C_cyc)
        both = np.concatenate(
            [legacy.astype(np.float32), cyc.astype(np.float32)], axis=1
        )
        data_stamp = both.reshape(origin_size + (-1,))
    else:
        raise ValueError(f"Unknown time encoding {timeenc}")
    return data_stamp.astype(np.float32)
