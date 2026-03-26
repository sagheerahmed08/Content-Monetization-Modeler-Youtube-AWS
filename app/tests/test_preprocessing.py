import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import fill_na_with_group_median, feature_engineering


def make_df(**kwargs):
    """Build a minimal test DataFrame, row count inferred from kwargs."""
    n = len(next(iter(kwargs.values()))) if kwargs else 3
    defaults = {
        "views": [1000, 2000, 0][:n] if n <= 3 else [1000] * n,
        "likes": [100, 200, 50][:n] if n <= 3 else [100] * n,
        "comments": [10, 20, 5][:n] if n <= 3 else [10] * n,
        "watch_time_minutes": [500.0, 800.0, np.nan][:n] if n <= 3 else [500.0] * n,
        "video_length_minutes": [10.0, 15.0, 10.0][:n] if n <= 3 else [10.0] * n,
        "subscribers": [5000, 5000, 1000][:n] if n <= 3 else [5000] * n,
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# --- fill_na_with_group_median ---

def test_fill_na_with_group_median_fills_nan():
    df = make_df()
    result = fill_na_with_group_median(df, "watch_time_minutes", ["video_length_minutes"])
    assert result["watch_time_minutes"].isna().sum() == 0


def test_fill_na_with_group_median_no_change_when_no_nan():
    df = make_df(watch_time_minutes=[500.0, 800.0, 300.0])
    original = df["watch_time_minutes"].copy()
    result = fill_na_with_group_median(df, "watch_time_minutes", ["video_length_minutes"])
    pd.testing.assert_series_equal(result["watch_time_minutes"], original)


def test_fill_na_with_group_median_correct_value():
    # Group [10.0] has values [500.0, NaN] → median = 500.0
    df = make_df()
    result = fill_na_with_group_median(df, "watch_time_minutes", ["video_length_minutes"])
    assert result["watch_time_minutes"].iloc[2] == 500.0


# --- feature_engineering ---

def test_feature_engineering_adds_columns():
    df = make_df()
    result = feature_engineering(df)
    assert "engagement_rate" in result.columns
    assert "avg_watch_time_per_view" in result.columns


def test_engagement_rate_correct():
    df = make_df(views=[1000], likes=[100], comments=[10], watch_time_minutes=[500.0])
    result = feature_engineering(df)
    expected = (100 + 10) / 1000
    assert pytest.approx(result["engagement_rate"].iloc[0], rel=1e-5) == expected


def test_avg_watch_time_correct():
    df = make_df(views=[1000], likes=[100], comments=[10], watch_time_minutes=[500.0])
    result = feature_engineering(df)
    expected = 500.0 / 1000
    assert pytest.approx(result["avg_watch_time_per_view"].iloc[0], rel=1e-5) == expected


def test_zero_views_no_division_error():
    """Views=0 should not raise; derived features should be 0."""
    df = make_df(views=[0], likes=[0], comments=[0], watch_time_minutes=[0.0])
    result = feature_engineering(df)
    assert result["engagement_rate"].iloc[0] == 0.0
    assert result["avg_watch_time_per_view"].iloc[0] == 0.0
