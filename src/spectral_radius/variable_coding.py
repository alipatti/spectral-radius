from collections.abc import Sequence
import polars as pl

from spectral_radius.constants import START_YEAR

# TODO: determine appropriate scaling for these


def ranks_at_t0(x: pl.Expr) -> pl.Expr:
    x_at_t0 = x.filter(year=pl.col("year").min())

    def ranks(xw: Sequence[pl.Series]) -> pl.Series:
        df = pl.DataFrame(xw)
        assert df.columns == ["x", "w"], df.columns

        rank = (
            pl.col("w")
            .cum_sum()
            .sub(0.5 * pl.col("w"))
            .truediv(pl.col("w").sum())
            .alias("rank")
        )
        return (
            df.group_by("x").agg(pl.col("w").sum()).sort("x").with_columns(rank)["rank"]
        )

    return x.replace_strict(
        old=x_at_t0.unique().sort(),
        new=pl.map_batches([x.alias("x"), "w"], ranks, return_dtype=pl.Float64),
    )


def unit_variance_at_t0(x: pl.Expr) -> pl.Expr:
    # TODO: weight everything in here

    # center
    x = x - x.mean()

    # divide by sd in initial period
    sd_at_t0 = x.filter(year=pl.col("year").min()).std()
    return x / sd_at_t0


def scale_to_pm_1(x: pl.Expr) -> pl.Expr:
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1
