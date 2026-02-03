from functools import cache
from itertools import combinations_with_replacement
from typing import Collection, Iterable, Mapping
import polars as pl
import numpy as np
import plotnine as pn
from plotnine_theme import theme_ali
from polars_utils.stats import mean
import inflection
from tqdm import tqdm

from spectral_radius.bootstrap import replicate_measures
from spectral_radius.constants import FIGURES, START_YEAR
from spectral_radius.gss import get_gss
from spectral_radius.gss import OPINION_CATEGORIES
from spectral_radius.gss.demographic_variables import DEMOGRAPHIC_VARIABLES
from spectral_radius.plot_helpers import (
    COLOR_SCALE,
    CATEGORY_WRAP,
    COLORS,
    PERCENT_CHANGE_SCALE,
    savefig,
)


def measures(
    df: pl.DataFrame,
    *,
    w: str = "w",
    columns: Collection[str],
    alpha: float | None = None,
) -> pl.DataFrame:
    # demean
    df = df.with_columns(pl.col(columns).pipe(lambda x: x - x.pipe(mean, w=w)))

    n = len(columns)
    sigma = np.empty((n, n))
    weights = df[w]

    # FIX: should this covariance matrix computation respect the sampling design?
    for (i, a), (j, b) in combinations_with_replacement(enumerate(columns), 2):
        product = df[b] * df[a] * weights

        total_weight = weights.filter(product.is_not_null()).sum()
        covariance = product.sum() / total_weight if total_weight > 0 else 0

        sigma[i, j] = covariance
        sigma[j, i] = covariance

    lambdas = np.linalg.eigh(sigma).eigenvalues  # eigenvalues

    measures = dict(
        # spectral radius i.e. l2 operator norm i.e. spectral norm
        rho=lambdas.max(),
        # total variance (i.e. trace/nuclear norm)
        trace=lambdas.sum(),
        # proportion exp. by first pc
        spectral_concentration=lambdas.max() / lambdas.sum(),
        # frobenius norm
        frob=np.linalg.norm(lambdas, ord=2),
        # total weight of group
        w=weights.sum(),
        # covariance matrix itself
        # sigma=sigma,
    )

    results = pl.DataFrame(measures)

    if not alpha:
        return results

    quantiles = dict(lo=alpha / 2, hi=1 - alpha / 2)
    confint = df.pipe(replicate_measures, columns, method="brr").select(
        pl.col("rho").quantile(q, interpolation="linear").alias(f"rho_{label}")
        for label, q in quantiles.items()
    )

    return pl.concat([results, confint], how="horizontal")


def measures_by_group(
    df: pl.DataFrame,
    variables: Collection[str],
    *,
    group: Collection[str] = ["year"],
    **kwargs,
):
    process_group = lambda df_of_group: (  # noqa
        # compute measures
        measures(df_of_group, columns=variables, **kwargs)
        # make sure col labels are still there
        .select(
            *(pl.lit(df_of_group[c].first()).alias(c) for c in group),
            pl.all(),
        )
    )

    # PERF: partition, joblib parallel map, concat
    return df.drop_nulls(group).group_by(*group).map_groups(process_group)


def measures_by_group_and_category(
    df: pl.DataFrame,
    categories: Mapping[str, Collection[str]],
    group: Collection[str],
    **kwargs,
):
    print("Calculating polarization measures")
    print(f" - Categories: {list(categories.keys())}")
    print(f" - Groups: {group}")

    return pl.concat(
        measures_by_group(df, vs, group=group, **kwargs).select(
            pl.lit(cat).alias("category"),
            pl.all(),
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
    categories: None | Iterable[str] = None,  # ["Welfare"],
    **kwargs,
):
    gss = get_gss().with_columns(binned_year(year_bin_width))

    categories = (
        {k: v for k, v in OPINION_CATEGORIES.items() if k in categories}
        if categories is not None
        else OPINION_CATEGORIES
    )

    rhos_pooled = gss.pipe(
        measures_by_group_and_category,
        categories,
        ["year"],
        **kwargs,
    )

    if not by:
        return rhos_pooled, pl.DataFrame(), pl.DataFrame()

    rhos_within = gss.pipe(
        measures_by_group_and_category,
        categories,
        ["year", by],
        **kwargs,
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
    rhos_pooled, *_ = polarization_over_time_data(year_bin_width=1, alpha=0.5)

    fig = (
        pn.ggplot(rhos_pooled, pn.aes("year", "rho"))
        + pn.geom_line()
        + pn.geom_ribbon(
            pn.aes(ymin="rho_lo", ymax="rho_hi"),
            alpha=0.1,
            fill=COLORS[1],
        )
        + CATEGORY_WRAP
        + theme_ali()
        + pn.labs(
            x="Year",
            y="Spectral Norm of Covariance Matrix",
        )
    )
    # fig.show()

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
    cuts = DEMOGRAPHIC_VARIABLES.keys()
    cuts = ("political_party",)
    cuts = ()

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
