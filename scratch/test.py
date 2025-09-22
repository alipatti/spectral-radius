"""
Possible measures:

Of individual variables:
 - variance
 - kurtosis (tail heaviness)
 - fourth central moment (var * kurt)

Of covariance matrix:
 - spectral gap (elongated data in one direction ... polarization?)
 - spectral radius (size of first eigenvalue)
 - "generalized variance" (det of cov matrix) total variance (trace) "generalized correlation" sqrt(1 - det / trace)
   (see https://math.stackexchange.com/questions/889425/)
   reduces to pearson when n = 2

Misc:
 - degree to which first component correlates with self-described party
 - proportion of party variance explained by responses
 - proportion of response variance explained by party
"""

from collections.abc import Collection
from typing import Callable, Iterable, Sequence
import polars as pl
import numpy as np
from polars_utils.smoothing import smooth
from polars_utils.covariance import covariance_matrix
import plotnine as pn

from src.data_prep import load_gss_data
from src.variables import subset_to_study


def demean(x: pl.Expr) -> pl.Expr:
    return x - x.mean()


def scale_to_pm_1(x: pl.Expr) -> pl.Expr:
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1


def moments(responses: pl.DataFrame):
    df = (
        responses.lazy()
        .select("year", *sum(subset_to_study.values(), start=[]))
        .unpivot(index=["year"])
        .with_columns(pl.col("value").pipe(scale_to_pm_1).over("year", "variable"))
        .group_by("year", "variable")
        .agg(
            kurtosis=pl.col("value").kurtosis(fisher=False),
            variance=pl.col("value").var(),
            mean=pl.col("value").mean(),
            fourth_moment=pl.col("value").kurtosis(fisher=False)
            * pl.col("value").var(),
        )
        .unpivot(
            index=["year", "variable"],
            variable_name="stat",
        )
        .collect()
    )

    (
        pn.ggplot(
            df.group_by("year", "stat").agg(pl.col("value").mean()),
            pn.aes("year", "value"),
        )
        + pn.geom_line()
        + pn.facet_wrap("stat", scales="free")
    ).show()

    (
        pn.ggplot(
            df.filter(pl.col("variable").is_in(subset_to_study["spending"])),
            pn.aes("year", "value", color="variable"),
        )
        + pn.geom_point()
        + pn.geom_line()
        + pn.facet_wrap("stat", scales="free")
    ).show()


def spectral_radius(X: np.ndarray):
    return np.linalg.eigvalsh(X).max()


def spectral_gap(X: np.ndarray, normalize_eigenvalues=True):
    eigvals = np.sort(np.linalg.eigvalsh(X))

    if normalize_eigenvalues:
        eigvals /= eigvals.sum()

    return eigvals[-1] - eigvals[-2]


def spectral_ratio(X: np.ndarray):
    eigvals = np.sort(np.linalg.eigvalsh(X))

    return eigvals[-1] / eigvals.sum()


def generalized_cov(X: np.ndarray):
    return np.linalg.det(X)


def mean_var(X: np.ndarray):
    return X.diagonal().mean()


def generalized_cor(X: np.ndarray):
    return np.sqrt(1 - np.linalg.det(X) / np.linalg.trace(X))


def first_pc(
    variables: Collection[str],
    w="wtssps",
) -> pl.Expr:
    def _first_pc(X: np.ndarray, w: np.ndarray):
        sigma = np.cov(X, rowvar=False, aweights=w)

        first_component = np.linalg.eigh(sigma).eigenvectors[-1]

        return pl.Series(X @ first_component)

    return pl.map_groups(
        [*variables, w],
        lambda series: _first_pc(np.column_stack(series[:-1]), w=series[-1].to_numpy()),
    ).alias("first_pc")


def covariance_summary_measures(
    responses: pl.DataFrame,
    variables: Collection[str],
    *,
    by=["year"],
    null_func: Callable[[pl.Expr], pl.Expr] = lambda x: x,
    preprocessing_func: Callable[[pl.Expr], pl.Expr] = scale_to_pm_1,
    summary_measures: Iterable[Callable[[np.ndarray], float]] = [
        generalized_cor,
        generalized_cov,
        mean_var,
        spectral_gap,
        spectral_radius,
        spectral_ratio,
    ],
    weight="wtssps",
):
    subset = (
        responses.select(*by, weight, *variables)
        .with_columns(pl.col(variables).pipe(null_func))
        .with_columns(pl.col(variables).pipe(preprocessing_func))
        .drop_nulls(variables)
    )

    covariance_matrices = {
        key: df.pipe(covariance_matrix, columns=pl.col(variables), w=weight)
        for key, df in subset.partition_by(
            *by, as_dict=True, maintain_order=True
        ).items()
    }

    return pl.from_records(
        [
            (*key, summary_measure.__name__, summary_measure(sigma))
            for summary_measure in summary_measures
            for key, sigma in covariance_matrices.items()
        ],
        schema=[*by, "measure", "value"],
        orient="row",
    )


def ballot_availability(responses: pl.DataFrame, year: int, variables: Sequence[str]):
    var_years = (
        responses.filter(year=year)
        .unpivot(on=variables, index="ballot")
        .group_by("ballot", "variable")
        .agg(non_missing=pl.col("value").is_not_null().mean().round_sig_figs(3))
    )

    return (
        pn.ggplot(var_years, pn.aes("ballot", "variable"))
        + pn.geom_tile(pn.aes(fill="non_missing"))
        + pn.geom_label(pn.aes(label="non_missing"))
    )


def plot_availability(responses: pl.DataFrame, variables: Sequence[str]):
    var_years = (
        responses.unpivot(on=variables, index=["year", "ballot"])
        .drop_nulls()
        .drop("value")
        .unique()
    )

    return (
        pn.ggplot(var_years, pn.aes("year", "variable"))
        + pn.geom_point()
        + pn.geom_line()
        + pn.facet_wrap("ballot")
    )


def main():
    responses, meta = load_gss_data(use_cache=True)
    responses = responses.filter(pl.col("year") >= 2000).cast({"ballot": str})

    variables = subset_to_study["reproductive"] + subset_to_study["sex"]
    variables = subset_to_study["spending"]
    variables = subset_to_study["race"]
    variables = subset_to_study["speech"]
    variables = subset_to_study["welfare"]

    # plot_availability(responses, variables).show()

    cov_measures = covariance_summary_measures(
        responses,
        variables=variables,
        by=["year", "ballot"],
    ).with_columns(
        pl.col("value")
        .pipe(smooth, "year", progress=True)
        .over("ballot", "measure")
        .alias("smooth"),
    )

    separate_ballots = (
        pn.ggplot(cov_measures, pn.aes("year", color="ballot"))
        + pn.geom_point(pn.aes(y="value"))
        + pn.geom_line(pn.aes(y="smooth"))
        + pn.facet_wrap("measure", scales="free")
    )

    pooled_ballots = (
        pn.ggplot(
            cov_measures.group_by("year", "measure").agg(pl.col("value").mean()),
            pn.aes("year"),
        )
        + pn.geom_point(pn.aes(y="value"))
        + pn.facet_wrap("measure", scales="free")
    )

    separate_ballots.show()
