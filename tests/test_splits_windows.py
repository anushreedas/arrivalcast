import numpy as np
import pandas as pd
import pytest

from arrivalcast.core.splits import chronological_split, get_chronological_windows
from arrivalcast.core.windows import compute_window


@pytest.mark.parametrize("n", [8, 24, 105, 731, 17544])
def test_split_is_contiguous_ordered_and_complete(n):
    s = chronological_split(n)
    allidx = np.concatenate([s.train, s.val, s.test])
    assert np.array_equal(allidx, np.arange(n))
    assert len(s.train) > 0 and len(s.val) > 0 and len(s.test) > 0


def test_tiny_sources_keep_a_test_event():
    for n in (2, 3, 6, 7):
        s = chronological_split(n)
        assert len(s.test) >= 1 and len(s.train) >= 1


def test_split_fractions_for_large_n():
    s = chronological_split(1000)
    assert (len(s.train), len(s.val), len(s.test)) == (700, 150, 150)


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        chronological_split(10, train_frac=0.9, val_frac=0.2)


def test_window_gaps_and_censored_tail():
    ts = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00", "2024-01-01 05:00"]))
    w = compute_window(ts, study_end=pd.Timestamp("2024-01-01 09:00"))
    assert np.allclose(w.gaps_hours, [2.0, 3.0])
    assert w.tail_hours == pytest.approx(4.0)
    assert w.total_hours == pytest.approx(9.0)


def test_split_windows_are_bounded_by_next_split(daily_timestamps):
    end = pd.Timestamp("2024-01-01")
    w = get_chronological_windows(daily_timestamps, global_study_end=end)
    ts = daily_timestamps.sort_values().reset_index(drop=True)
    # train ends where validation begins; the test split ends at the global study end
    assert w["train"].window.study_end == w["val"].timestamps.iloc[0]
    assert w["val"].window.study_end == w["test"].timestamps.iloc[0]
    assert w["test"].window.study_end == end
    # no event appears in two splits
    n = sum(len(w[k].timestamps) for k in ("train", "val", "test"))
    assert n == len(ts)
