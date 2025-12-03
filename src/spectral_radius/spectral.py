from functools import cache
from typing import Collection, Mapping
import polars as pl
import numpy as np
from polars_utils.covariance import covariance_matrix
import plotnine as pn
from plotnine_theme import theme_ali
from polars_utils.stats import mean
import inflection
from tqdm import tqdm

from spectral_radius.constants import FIGURES, START_YEAR
from spectral_radius.gss import get_gss
from spectral_radius.gss import OPINION_CATEGORIES, DEMOGRAPHIC_VARIABLES
from spectral_radius.plot_helpers import (
    COLOR_SCALE,
    CATEGORY_WRAP,
    PERCENT_CHANGE_SCALE,
    savefig,
)


def measures(
    group: pl.DataFrame,
    *,
    w="w",
    columns: Collection[str],
) -> pl.DataFrame:
    # PERF: speed up this computation (very slow rn)
    sigma = covariance_matrix(
        group,
        w=w,
        columns=columns,
        nulls="pairwise_complete",
    )

    lambdas = np.linalg.eigh(sigma).eigenvalues  # eigenvalues

    measures = dict(
        # spectral radius i.e. l2 operator norm i.e. spectral norm
        rho=lambdas.max(),
        # total variance (i.e. trace/nuclear norm)
        trace=lambdas.sum(),
        # proportion exp. by first pc
        spectral_concentration=lambdas.max() / lambdas.sum(),
        # spectral gap
        spectral_gap=lambdas[-1] - lambdas[-2],
        # frobenius norm
        frob=np.linalg.norm(lambdas, ord=2),
        # total weight of group
        w=group.select(pl.col(w).sum()).item(),
        # covariance matrix itself
        # sigma=sigma,
    )

    return pl.DataFrame(measures)


def measures_by_group(
    df: pl.DataFrame,
    variables: Collection[str],
    *,
    group: Collection[str] = ["year"],
):
    return (
        df.drop_nulls(group)
        .group_by(*group)
        .map_groups(
            lambda df_grouped: (
                # compute measures
                measures(df_grouped, columns=variables)
                # make sure col labels are still there
                .select(
                    *(pl.lit(df_grouped.get_column(c).first()).alias(c) for c in group),
                    pl.all(),
                )
            )
        )
    )


# PERF: cache this?
def measures_by_group_and_category(
    df: pl.DataFrame,
    categories: Mapping[str, Collection[str]],
    group: Collection[str],
):
    print("Calculating polarization measures")
    print(f" - Categories: {list(categories.keys())}")
    print(f" - Groups: {group}")

    return pl.concat(
        measures_by_group(df, vs, group=group).select(
            pl.lit(cat).alias("category"), pl.all()
        )
        for cat, vs in tqdm(categories.items())
    ).sort("*")


def binned_year(bin_size: int | None = 1) -> pl.Expr:
    return (
        pl.col("year")
        .sub(START_YEAR)
        .floordiv(bin_size)
        .mul(bin_size)
        .add(START_YEAR + pl.lit(bin_size) / 2)
    )


@cache
def polarization_over_time_data(
    by: str | None = None,
    metric="rho",
    year_bin_width: int | None = 5,
):
    gss = get_gss().with_columns(binned_year(year_bin_width))

    rhos_pooled = gss.pipe(
        measures_by_group_and_category,
        OPINION_CATEGORIES,
        ["year"],
    )

    if not by:
        return rhos_pooled, pl.DataFrame(), pl.DataFrame()

    rhos_within = gss.pipe(
        measures_by_group_and_category,
        OPINION_CATEGORIES,
        ["year", by],
    )

    group_decomposition = (
        rhos_pooled.select("year", "category", metric)
        .join(
            rhos_within.group_by("year", "category").agg(
                pl.col(metric).pipe(mean, w="w")
            ),
            on=["year", "category"],
            suffix="_within",
        )
        .with_columns(
            # calculate slack term
            pl.col(metric).sub(f"{metric}_within").alias(f"{metric}_between"),
        )
    )

    return rhos_pooled, rhos_within, group_decomposition


def polarization_figure() -> pn.ggplot:
    rhos_pooled, *_ = polarization_over_time_data(year_bin_width=1)

    fig = (
        pn.ggplot(rhos_pooled, pn.aes("year", "rho"))
        + pn.geom_line()
        + CATEGORY_WRAP
        + theme_ali()
        + pn.labs(
            x="Year",
            y="Spectral Norm of Covariance Matrix",
        )
    )

    return fig


def trace_decomp_figure() -> pn.ggplot:
    rhos_pooled, *_ = polarization_over_time_data()

    renames = {
        "spectral_concentration": "Spectral Concentration",
        "trace": "Total Variance",
    }

    decomposition = (
        rhos_pooled.sort("*")
        .select(
            "category",
            "year",
            pl.col("rho", "trace", "spectral_concentration").pipe(
                lambda x: x / x.first().over("category")
            ),
        )
        .unpivot(index=["year", "category", "rho"])
        .with_columns(pl.col("variable").replace_strict(renames))
        .pipe(pn.ggplot, pn.aes("year"))
        + pn.geom_line(pn.aes(y="rho"), linetype="dashed", color="gray")
        + pn.geom_line(pn.aes(y="value", color="variable"))
        + CATEGORY_WRAP
        + theme_ali()
        + COLOR_SCALE
        + PERCENT_CHANGE_SCALE
        + pn.labs(
            x="Year",
            color="Component",
        )
    )

    return decomposition


def subgroup_polarization_figure(by: str) -> pn.ggplot:
    _, rhos_within, group_decomposition = polarization_over_time_data(by)

    by_group = (
        rhos_within.with_columns(
            # TODO: do we want to demean?
            # pl.col(metric).pipe(lambda x: (x / x.log().mean().exp()).over("category"))
        ).pipe(
            pn.ggplot,
            pn.aes("year", "rho"),
        )
        + pn.geom_line(data=group_decomposition, linetype="dashed", color="gray")
        + pn.geom_line(pn.aes(color=by))
        + CATEGORY_WRAP
        + COLOR_SCALE
        + theme_ali()
        + pn.labs(
            x="Year",
            y="Spectral Norm of Covariance Matrix",
            color=inflection.titleize(by),
        )
    )

    return by_group


def group_decomp_figure(by: str = "race") -> pn.ggplot:
    _, _, group_decomposition = polarization_over_time_data(by)

    # counterfactual if we fix the other and ONLY allow rho/between to vary
    residual_at_start = pl.col("rho").sub("value").first().over("category", "variable")
    counterfactual = pl.col("value").add(residual_at_start).alias("counterfactual")

    group = inflection.titleize(by)
    labeled_component = pl.col("variable").replace_strict(
        {"rho_within": f"Within {group}", "rho_between": f"Between {group}"}
    )

    decomposition = (
        pn.ggplot(
            group_decomposition.unpivot(index=["year", "category", "rho"])
            .sort("*")
            .with_columns(labeled_component, counterfactual)
            .with_columns(
                # scale to percentage change
                pl.col("rho", "counterfactual").pipe(
                    lambda x: x / x.first().over("category", "variable")
                )
            ),
            pn.aes("year"),
        )
        + pn.geom_line(pn.aes(y="counterfactual", color="variable"))
        + pn.geom_line(pn.aes(y="rho"), linetype="dashed", color="gray")
        + CATEGORY_WRAP
        + COLOR_SCALE
        + PERCENT_CHANGE_SCALE
        + theme_ali()
        + pn.labs(
            x="Year",
            color="Component",
        )
    )

    return decomposition


def main():
    # cuts = DEMOGRAPHIC_VARIABLES.keys()
    cuts = ("political_party",)
    # cuts = ()

    all_figures = (
        {
            "pooled/pooled": polarization_figure(),
            "decompositions/trace_concentration": trace_decomp_figure(),
        }
        | {f"decompositions/{group}": group_decomp_figure(group) for group in cuts}
        | {f"by_group/{group}": subgroup_polarization_figure(group) for group in cuts}
    )

    for path_root, fig in all_figures.items():
        savefig(fig, FIGURES / "gss" / path_root, size=(8.5 - 2, 11 - 3.5))


if __name__ == "__main__":
    main()
