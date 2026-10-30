from collections.abc import Collection, Mapping
from functools import cache
from itertools import chain, pairwise

import inflection
import numpy as np
import plotnine as pn
import polars as pl
from plotnine_theme import theme_ali
from polars import selectors as cs
from polars_utils.stats import mean

from spectral_radius.analysis import analysis_frame, survey_data
from spectral_radius.constants import FIGURES, START_YEAR
from spectral_radius.plot_helpers import (
    CATEGORY_WRAP,
    COLOR_SCALE,
    PERCENT_CHANGE_SCALE,
    savefig,
)
from spectral_radius.surveys import DEMOGRAPHICS, QUESTION_SETS


def pairwise_complete_covariance(
    x: np.ndarray,
    observed: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Weighted covariance of the (already demeaned) columns of `x`, pairwise-complete.

    Entry (a, b) is the sum of `w * x_a * x_b` over rows where both a and b are
    observed, divided by the sum of `w` over exactly those rows (0 if that is 0).
    Unobserved entries of `x` must be 0 so they drop out of the numerator.
    """
    observed = observed.astype(float)

    numerator = (x * w[:, None]).T @ x
    total_weight = (observed * w[:, None]).T @ observed

    return np.divide(
        numerator,
        total_weight,
        out=np.zeros_like(numerator),
        where=total_weight > 0,
    )


def spectral_measures(sigma: np.ndarray) -> dict[str, float]:
    lambdas = np.linalg.eigh(sigma).eigenvalues  # eigenvalues

    return {
        # spectral radius i.e. l2 operator norm i.e. spectral norm
        "rho": lambdas.max(),
        # total variance (i.e. trace/nuclear norm)
        "trace": lambdas.sum(),
        # proportion exp. by first pc
        "spectral_concentration": lambdas.max() / lambdas.sum(),
        # spectral gap
        "spectral_gap": lambdas[-1] - lambdas[-2],
        # frobenius norm
        "frob": np.linalg.norm(lambdas, ord=2),
    }


def measures_by_group_and_category(
    df: pl.DataFrame,
    categories: Mapping[str, Collection[str]],
    group: Collection[str],
    *,
    w: str = "w",
) -> pl.DataFrame:
    """
    Spectral measures of each group's covariance matrix of each category's variables.

    Each variable is demeaned (unweighted) over its non-null rows in the group,
    then covariances are weighted and pairwise-complete.
    """
    print("Calculating polarization measures")
    print(f" - Categories: {list(categories.keys())}")
    print(f" - Groups: {group}")

    variables = list(dict.fromkeys(chain.from_iterable(categories.values())))
    position = {v: i for i, v in enumerate(variables)}
    category_columns = {
        cat: np.array([position[v] for v in vs]) for cat, vs in categories.items()
    }

    # sorted so each group's rows are contiguous
    demeaned = (
        df.drop_nulls(group)
        .select(
            *group,
            pl.col(w),
            pl.col(variables).pipe(lambda x: x - x.mean()).over(group),
        )
        .sort(group, maintain_order=True)
    )

    groups = (
        demeaned.group_by(group, maintain_order=True)
        .agg(pl.len().alias("n"), pl.col(w).sum())
        # as strings, matching the python literals the labels used to be built from
        .with_columns((cs.categorical() | cs.enum()).cast(pl.String))
        .with_row_index("group_index")
    )
    group_ends = groups.get_column("n").cum_sum().to_list()
    group_bounds = enumerate(pairwise([0, *group_ends]))

    # a null weight makes the product null in every pair, i.e. contributes nothing
    x = demeaned.select(variables).fill_null(0).to_numpy()
    observed = demeaned.select(pl.col(variables).is_not_null()).to_numpy()
    weights = demeaned.get_column(w).fill_null(0).to_numpy()

    spectra = pl.DataFrame(
        dict(
            group_index=i,
            category=cat,
            **spectral_measures(
                pairwise_complete_covariance(
                    x[start:end, columns],
                    observed[start:end, columns],
                    weights[start:end],
                )
            ),
        )
        for i, (start, end) in group_bounds
        for cat, columns in category_columns.items()
    ).with_columns(pl.col("group_index").cast(pl.UInt32))

    return (
        spectra.join(groups, on="group_index", validate="m:1")
        .select(
            "category", *group, pl.exclude("category", *group, "group_index", "n", w), w
        )
        .sort("*")
    )


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
    survey: str = "gss",
):
    # undeclared demographics are null throughout, so grouping by one is an error
    if by:
        survey_data().require_demographic(survey, by)

    responses = analysis_frame(survey).with_columns(binned_year(year_bin_width))
    categories = QUESTION_SETS[survey]

    if not by:
        rhos_pooled = responses.pipe(
            measures_by_group_and_category,
            categories,
            ["year"],
        )
        return rhos_pooled, pl.DataFrame(), pl.DataFrame()

    # reuse the cached pooled measures rather than recomputing them for every `by`
    rhos_pooled, *_ = polarization_over_time_data(
        year_bin_width=year_bin_width,
        survey=survey,
    )

    rhos_within = responses.pipe(
        measures_by_group_and_category,
        categories,
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


def polarization_figure(survey: str = "gss") -> pn.ggplot:
    rhos_pooled, *_ = polarization_over_time_data(year_bin_width=1, survey=survey)

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


def trace_decomp_figure(survey: str = "gss") -> pn.ggplot:
    rhos_pooled, *_ = polarization_over_time_data(survey=survey)

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


def subgroup_polarization_figure(by: str, survey: str = "gss") -> pn.ggplot:
    _, rhos_within, group_decomposition = polarization_over_time_data(
        by,
        survey=survey,
    )

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


def group_decomp_figure(by: str = "race", survey: str = "gss") -> pn.ggplot:
    _, _, group_decomposition = polarization_over_time_data(by, survey=survey)

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


def main(survey: str = "gss"):
    provided = survey_data().demographics[survey]
    demographics = [d for d in DEMOGRAPHICS if d in provided]

    all_figures = (
        {
            "pooled/pooled": polarization_figure(survey),
            "decompositions/trace_concentration": trace_decomp_figure(survey),
        }
        | {
            f"decompositions/{group}": group_decomp_figure(group, survey)
            for group in demographics
        }
        | {
            f"by_group/{group}": subgroup_polarization_figure(group, survey)
            for group in demographics
        }
    )

    for path_root, fig in all_figures.items():
        savefig(fig, FIGURES / survey / path_root, size=(8.5 - 2, 11 - 3.5))


if __name__ == "__main__":
    main()
