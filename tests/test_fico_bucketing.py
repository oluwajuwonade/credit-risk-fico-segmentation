import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fico_bucketing import fit_fico_buckets, assign_rating


def _toy_dataset(n=6000, seed=0):
    """Small synthetic set where default risk is a clean, monotonic
    function of FICO score, used to sanity-check the algorithm itself
    (separately from the main project's noisier synthetic portfolio)."""
    rng = np.random.default_rng(seed)
    fico = rng.integers(300, 851, size=n)
    p_default = np.clip(1 - (fico - 300) / 550, 0.01, 0.99)
    default = rng.binomial(1, p_default)
    return pd.Series(fico), pd.Series(default)


def test_returns_requested_number_of_buckets():
    fico, default = _toy_dataset()
    boundaries, summary = fit_fico_buckets(fico, default, n_buckets=5, min_bucket_size=100)
    assert len(summary) == 5
    assert len(boundaries) == 4


def test_buckets_partition_all_borrowers():
    fico, default = _toy_dataset()
    _, summary = fit_fico_buckets(fico, default, n_buckets=6, min_bucket_size=100)
    assert summary["n_borrowers"].sum() == len(fico)


def test_default_rate_is_monotonic_in_fico_for_clean_signal():
    # With a clean, strong monotonic risk signal, log-likelihood-optimal
    # bucketing should recover a monotonic default-rate progression.
    fico, default = _toy_dataset(n=20000)
    _, summary = fit_fico_buckets(fico, default, n_buckets=6, min_bucket_size=500)
    summary = summary.sort_values("fico_min")
    assert summary["default_rate"].is_monotonic_decreasing


def test_assign_rating_matches_bucket_range():
    fico, default = _toy_dataset()
    _, summary = fit_fico_buckets(fico, default, n_buckets=5, min_bucket_size=100)
    for _, row in summary.iterrows():
        mid = (row["fico_min"] + row["fico_max"]) // 2
        assert assign_rating(mid, summary) == row["rating"]


def test_assign_rating_clamps_out_of_range_scores():
    fico, default = _toy_dataset()
    _, summary = fit_fico_buckets(fico, default, n_buckets=5, min_bucket_size=100)
    low_rating = assign_rating(1, summary)  # below any real FICO score
    high_rating = assign_rating(999, summary)  # above any real FICO score
    assert low_rating == summary.sort_values("fico_min").iloc[0]["rating"]
    assert high_rating == summary.sort_values("fico_min").iloc[-1]["rating"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
