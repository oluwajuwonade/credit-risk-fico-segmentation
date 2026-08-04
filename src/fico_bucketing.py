"""
fico_bucketing.py

Implements log-likelihood-optimal bucketing of FICO scores into a fixed
number of rating buckets, solved via dynamic programming.

Why this approach (vs. equal-width or equal-frequency binning):
Equal-width/equal-frequency bins ignore where the *default rate* actually
changes. Log-likelihood-optimal bucketing instead treats each bucket's
defaults as Binomial(n_i, p_i) and chooses bucket boundaries that maximize
the total log-likelihood of the observed defaults given the buckets, i.e.
boundaries that best separate genuinely different-risk borrowers.

This mirrors the bucketing task in JPMorgan Chase's "Quantitative Research"
Forage simulation, re-implemented from scratch for a general/synthetic
dataset here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x * log(y), defined as 0 when x == 0 (standard 0*log(0)=0 convention)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(x == 0, 0.0, x * np.log(np.where(y == 0, 1.0, y)))
    return out


def _bucket_log_likelihood(n: int, k: int) -> float:
    """Log-likelihood of k defaults out of n borrowers under a single
    bucket-level default probability p = k / n (its own MLE)."""
    if n == 0:
        return 0.0
    p = k / n
    ll = _safe_xlogy(np.array([k]), np.array([p]))[0]
    ll += _safe_xlogy(np.array([n - k]), np.array([1 - p]))[0]
    return float(ll)


def fit_fico_buckets(
    fico_scores: pd.Series,
    defaults: pd.Series,
    n_buckets: int = 10,
    min_bucket_size: int = 50,
) -> tuple[list[int], pd.DataFrame]:
    """
    Find the log-likelihood-optimal boundaries splitting FICO scores into
    `n_buckets` contiguous ranges.

    Parameters
    ----------
    fico_scores     : integer FICO scores
    defaults        : 0/1 default indicator, same index/order as fico_scores
    n_buckets       : number of rating buckets to produce
    min_bucket_size : minimum borrowers required in a bucket. Pure
                      log-likelihood maximization is prone to carving out
                      tiny, noisy buckets in sparse regions of the score
                      distribution (e.g. a bucket of 3 borrowers with a
                      100% default rate); this constraint keeps every
                      bucket large enough for its default rate to be a
                      meaningful estimate.

    Returns
    -------
    boundaries : sorted list of FICO cut points (upper edge of each bucket
                 except the last, which is open-ended upward)
    summary    : DataFrame with one row per bucket (rating, fico range,
                 n borrowers, n defaults, default rate)
    """
    df = pd.DataFrame({"fico": fico_scores, "default": defaults})
    grouped = (
        df.groupby("fico")["default"]
        .agg(n="count", k="sum")
        .sort_index()
    )
    scores = grouped.index.to_numpy()
    n_arr = grouped["n"].to_numpy()
    k_arr = grouped["k"].to_numpy()
    U = len(scores)  # number of unique FICO values

    # Prefix sums so any range [a, b) is O(1) to query
    N_cum = np.concatenate([[0], np.cumsum(n_arr)])
    K_cum = np.concatenate([[0], np.cumsum(k_arr)])

    def range_ll(a: int, b: int) -> float:
        n = int(N_cum[b] - N_cum[a])
        k = int(K_cum[b] - K_cum[a])
        return _bucket_log_likelihood(n, k)

    # Precompute all range log-likelihoods once: cost[a][b] for a < b <= U.
    # Ranges with fewer than min_bucket_size borrowers are disallowed (-inf)
    # so the optimizer can't exploit sparse/noisy tail regions.
    cost = np.full((U + 1, U + 1), -np.inf)
    for a in range(U):
        for b in range(a + 1, U + 1):
            n = int(N_cum[b] - N_cum[a])
            if n >= min_bucket_size:
                cost[a, b] = range_ll(a, b)

    # DP: dp[j][i] = best total log-likelihood splitting scores[0:i] into j buckets
    NEG_INF = -np.inf
    dp = np.full((n_buckets + 1, U + 1), NEG_INF)
    choice = np.zeros((n_buckets + 1, U + 1), dtype=int)
    dp[0, 0] = 0.0

    for j in range(1, n_buckets + 1):
        for i in range(j, U + 1):
            best_val, best_a = NEG_INF, -1
            # bucket j covers group-index range [a, i)
            for a in range(j - 1, i):
                if dp[j - 1, a] == NEG_INF:
                    continue
                val = dp[j - 1, a] + cost[a, i]
                if val > best_val:
                    best_val, best_a = val, a
            dp[j, i] = best_val
            choice[j, i] = best_a

    # Backtrack to recover boundaries
    cuts = []
    i, j = U, n_buckets
    while j > 0:
        a = choice[j, i]
        cuts.append(a)
        i, j = a, j - 1
    cuts = sorted(cuts)[1:]  # drop the leading 0

    boundaries = [int(scores[c]) for c in cuts]  # upper (exclusive) edge scores

    # Build a human-readable summary table
    edges = [scores[0]] + boundaries + [scores[-1] + 1]
    rows = []
    for rating, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        mask = (scores >= lo) & (scores < hi)
        n = int(n_arr[mask].sum())
        k = int(k_arr[mask].sum())
        rows.append(
            {
                "rating": rating,  # 1 = best credit quality by convention here
                "fico_min": int(lo),
                "fico_max": int(hi - 1),
                "n_borrowers": n,
                "n_defaults": k,
                "default_rate": round(k / n, 4) if n else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    # Re-rank so rating 1 = LOWEST default rate (best quality), ascending risk
    summary = summary.sort_values("default_rate").reset_index(drop=True)
    summary["rating"] = summary.index + 1
    summary = summary.sort_values("fico_min").reset_index(drop=True)
    return boundaries, summary


def assign_rating(fico_score: int, summary: pd.DataFrame) -> int:
    """Map a single FICO score to its rating bucket using a fitted summary table."""
    row = summary[(summary["fico_min"] <= fico_score) & (fico_score <= summary["fico_max"])]
    if row.empty:
        # Clamp to nearest bucket for out-of-range scores
        if fico_score < summary["fico_min"].min():
            return int(summary.iloc[0]["rating"])
        return int(summary.iloc[-1]["rating"])
    return int(row.iloc[0]["rating"])
