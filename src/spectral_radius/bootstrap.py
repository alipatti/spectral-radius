from typing import Iterator, Literal

from plotnine_theme import theme_ali
import polars as pl
import scipy.linalg
import numpy as np
import plotnine as pn
from scipy.stats import norm, normaltest, multinomial
from tqdm import tqdm

from spectral_radius.gss.opinion_variables import OPINION_CATEGORIES
from spectral_radius.plot_helpers import COLOR_SCALE, COLORS
from spectral_radius.spectral import get_gss, measures

ResamplingMethod = Literal["brr", "bootstrap"]


def main(
    years=range(2000, 2011, 4),
    categories=("Welfare",),
    methods: list[ResamplingMethod] = ["brr", "bootstrap"],
):
    resamples = pl.concat(
        get_gss()
        .filter(pl.col("year") == year)
        .pipe(replicate_measures, OPINION_CATEGORIES[category], method=method)
        .with_columns(
            category=pl.lit(category),
            year=pl.lit(year),
            method=pl.lit(method),
        )
        for category in categories
        for year in years
        for method in methods
    )

    (
        pn.ggplot(resamples, pn.aes("rho", color="method"))
        + pn.geom_density()
        + pn.facet_wrap("year", scales="free")
    ).show()

    p = (
        pl.col("rho")
        .map_batches(
            lambda x: normaltest(x).pvalue,
            returns_scalar=True,
        )
        .round(2)
        .over("year", "category")
    )
    densities = resamples.with_columns(
        # density
        pl.col("rho")
        .map_batches(lambda x: norm.pdf(x, loc=x.mean(), scale=x.std()))
        .over("year", "category")
        .alias("normal_density"),
        group=pl.format("{}, {}\n$p={}$", "category", "year", p),
    )

    alpha = 0.05
    quantiles = (alpha / 2, 1 - alpha / 2)
    empirical_ci = pl.concat_arr(pl.col("rho").quantile(q) for q in quantiles)
    normal_ci = (
        pl.col("rho")
        .map_batches(
            lambda x: norm.ppf(quantiles, loc=x.mean(), scale=x.std()),
        )
        .implode()
        .list.to_array(2)
    )

    confidence_intervals = (
        resamples.group_by("year", "category")
        .agg(normal=normal_ci, empirical=empirical_ci)
        .unpivot(index=["year", "category"], variable_name="method")
        .explode("value")
        .join(densities.select("year", "category", "group"), ["year", "category"])
    )

    p = (
        pn.ggplot(densities, pn.aes(x="rho"))
        + pn.geom_rug()
        + theme_ali()
        + COLOR_SCALE
        + pn.facet_wrap("group", scales="free")
        # CIs
        + pn.geom_vline(
            pn.aes(xintercept="value", color="method"),
            data=confidence_intervals,
            inherit_aes=False,
            # linetype="dashed",
        )
        # densities
        + pn.geom_density(color=COLORS[0], linetype="dashed")
        + pn.geom_line(pn.aes(y="normal_density"), color=COLORS[1], linetype="dashed")
        + pn.labs(color="Method", x="Spectral Radius", y="Density")
    )
    p.show()


def replicate_measures(
    df: pl.DataFrame,
    cols: list[str],
    *,
    method: ResamplingMethod = "brr",
    **kwargs,
):
    if method == "brr":
        weights = brr_weights
    elif method == "bootstrap":
        weights = bootstrap_weights
    else:
        raise ValueError

    return pl.concat(
        df.with_columns(w)
        .pipe(measures, columns=cols)
        .with_columns(
            bootstrap_iter=i,
        )
        for i, w in enumerate(weights(df, **kwargs))
    )


def bootstrap_weights(
    df: pl.DataFrame,
    *,
    weight_col=pl.col("w"),
    n_iters=150,
) -> Iterator[pl.Expr]:

    np.random.seed(1280)  # scipy uses numpy for rng

    spine = df.select("vstrat", "vpsu").unique()

    for _ in tqdm(range(n_iters)):
        # want mapping vstrat, vpsu -> weight
        # where within-strat weights are dirichlet/multinomial
        weights = spine.with_columns(
            pl.repeat(1, pl.len())
            .map_batches(
                lambda ones: multinomial.rvs(ones.len(), ones.to_numpy() / ones.len()),
                return_dtype=pl.Float64,
            )
            .over("vstrat")
            .alias("weight")
        ).sort("vstrat")

        yield weight_col * pl.struct("vstrat", "vpsu").replace_strict(
            weights.select("vstrat", "vpsu").to_struct(),
            weights["weight"],
        )


def brr_weights(
    df: pl.DataFrame,
    *,
    strata_col=pl.col("vstrat"),
    unit_col=pl.col("vpsu"),
    weight_col=pl.col("w"),
) -> Iterator[pl.Expr]:
    """
    https://documentation.sas.com/doc/en/statug/15.2/statug_surveyphreg_details29.htm
    https://en.wikipedia.org/wiki/Balanced_repeated_replication
    """
    strata = df["vstrat"].unique()

    n_strata = strata.len()

    # smallest power of 2 greater than H
    n_replicates = 2 ** np.ceil(np.log2(n_strata))
    hadamard_matrix = scipy.linalg.hadamard(n_replicates)[:, :n_strata]

    # convert +1/-1 -> 1, 2
    psu_replicates = (hadamard_matrix + 1) / 2 + 1

    for selected_psu in tqdm(psu_replicates):
        row_is_in_selected_psu = unit_col == strata_col.replace_strict(
            strata,
            selected_psu,
        )

        yield weight_col * pl.when(row_is_in_selected_psu).then(2).otherwise(0)
