"""
Encodings of survey responses as numbers, for `wide`.

Each maps a variable's answers, as their positions on its scale
(`0` to `n_levels - 1`, in scale order; one column over the rows of the wide frame),
and the scale's number of levels to analysis values.
"""

from collections.abc import Callable

import polars as pl

type Encoding = Callable[[pl.Expr, int], pl.Expr]

# TODO: determine appropriate scaling for these


def unit_variance_at_t0(x: pl.Expr, _n_levels: int) -> pl.Expr:
    # TODO: weight everything in here

    # center
    x = x - x.mean()

    # divide by sd in initial period
    sd_at_t0 = x.filter(year=pl.col("year").min()).std()
    return x / sd_at_t0


def positions_to_pm_1(x: pl.Expr, n_levels: int) -> pl.Expr:
    """Space the scale's levels evenly over [-1, 1], from its first to its last."""
    return x / (n_levels - 1) * 2 - 1


def scale_observed_range_to_pm_1(x: pl.Expr, _n_levels: int) -> pl.Expr:
    """Scale the range observed over the wide frame's rows onto [-1, 1]."""
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1
