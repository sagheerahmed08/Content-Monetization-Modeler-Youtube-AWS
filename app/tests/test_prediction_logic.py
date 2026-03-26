import numpy as np
import pandas as pd
import pytest

# --- Validation logic (mirrors 4_Prediction.py) ---

def get_validation_errors(views, likes, comments, watch_time, video_length):
    """Pure replication of the prediction form validation rules."""
    errors = []
    if views == 0:
        errors.append("Views must be greater than 0.")
    if likes > views:
        errors.append("Likes cannot exceed Views.")
    if comments > views:
        errors.append("Comments cannot exceed Views.")
    if watch_time > views * video_length:
        errors.append("Watch Time cannot exceed Views × Video Length.")
    return errors


def apply_revenue_cap(raw_pred, views, max_rpm=20.0):
    """Pure replication of the revenue cap logic."""
    cap = views * max_rpm / 1000
    return float(np.clip(raw_pred, 0.0, cap))


def compute_derived_features(views, likes, comments, watch_time):
    """Pure replication of engagement_rate and avg_watch_time_per_view."""
    safe_views = views if views != 0 else 1
    engagement_rate = (likes + comments) / safe_views
    avg_watch_time = watch_time / safe_views
    return engagement_rate, avg_watch_time


# --- Validation tests ---

def test_zero_views_is_invalid():
    assert "Views must be greater than 0." in get_validation_errors(0, 0, 0, 0, 10)

def test_valid_inputs_no_errors():
    assert get_validation_errors(1000, 100, 10, 500, 10) == []

def test_likes_exceeding_views():
    errors = get_validation_errors(100, 200, 5, 50, 10)
    assert "Likes cannot exceed Views." in errors

def test_comments_exceeding_views():
    errors = get_validation_errors(100, 50, 200, 50, 10)
    assert "Comments cannot exceed Views." in errors

def test_watch_time_exceeding_max():
    # 100 views × 10 min = 1000 max; 1500 > 1000
    errors = get_validation_errors(100, 50, 5, 1500, 10)
    assert "Watch Time cannot exceed Views × Video Length." in errors

def test_watch_time_exactly_at_limit_is_valid():
    # exactly at limit: 100 × 10 = 1000
    assert get_validation_errors(100, 50, 5, 1000, 10) == []

def test_multiple_errors_returned():
    errors = get_validation_errors(100, 200, 200, 9999, 10)
    assert len(errors) == 3  # likes, comments, watch_time all invalid


# --- Revenue cap tests ---

def test_cap_clamps_large_prediction():
    # 1 view × $20/1000 = $0.02 cap
    assert apply_revenue_cap(280.0, 1) == pytest.approx(0.02, rel=1e-5)

def test_cap_does_not_clamp_reasonable_prediction():
    # 10_000 views → cap = $200; pred = $50 is fine
    assert apply_revenue_cap(50.0, 10_000) == pytest.approx(50.0, rel=1e-5)

def test_cap_clamps_negative_to_zero():
    assert apply_revenue_cap(-5.0, 1000) == 0.0

def test_cap_exactly_at_limit():
    cap = 1000 * 20.0 / 1000  # = $20.0
    assert apply_revenue_cap(20.0, 1000) == pytest.approx(20.0, rel=1e-5)


# --- Derived feature tests ---

def test_engagement_rate_formula():
    er, _ = compute_derived_features(1000, 100, 50, 500)
    assert er == pytest.approx(150 / 1000, rel=1e-5)

def test_avg_watch_time_formula():
    _, awt = compute_derived_features(1000, 100, 50, 500)
    assert awt == pytest.approx(500 / 1000, rel=1e-5)

def test_zero_views_no_division_error():
    er, awt = compute_derived_features(0, 0, 0, 0)
    assert er == 0.0
    assert awt == 0.0
