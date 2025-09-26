import matplotlib as mpl
from plotnine.facets import labeller
import polars as pl
import numpy as np
from polars_utils.covariance import covariance_matrix
import plotnine as pn
from plotnine_theme import theme_ali

from spectral_radius.gss import load_gss_data
from spectral_radius.gss.variables import VARIABLE_CATEGORIES, ALL_VARIABLES
from spectral_radius.variable_coding import scale_to_pm_1


mpl.use("pgf")
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,  # do not override with mpl defaults
    }
)


def spectral_radius(X: np.ndarray) -> float:
    lambdas = np.linalg.eigh(X).eigenvalues[::-1]

    return lambdas[0]


def measures(df: pl.DataFrame, *, w="w", columns: list[str]) -> pl.DataFrame:
    X = covariance_matrix(
        df,
        w=w,
        columns=columns,
        nulls="pairwise_complete",
    )

    lambdas = np.linalg.eigh(X).eigenvalues  # eigenvalues
    rho = lambdas[-1]  # spectral radius

    measures = dict(
        rho=rho,
        trace=X.trace(),  # total variance
        proportion=rho / X.trace(),  # proportion exp. by first pc
        gap=lambdas[-1] - lambdas[-2],  # spectral gap
    )

    return pl.DataFrame(measures)


def main():
    df = (
        load_gss_data(use_cache=True)[0]
        .select(
            "year",
            pl.col("wtssps").alias("w"),
            pl.col(ALL_VARIABLES).pipe(scale_to_pm_1),
        )
        .filter(pl.col("year") >= 1990)
    )

    rhos = pl.concat(
        df.group_by("year")
        .map_groups(
            lambda df: measures(
                df,
                columns=variables,
            ).select(df["year"].unique(), pl.lit(category).alias("category"), pl.all())
        )
        .sort("year")
        for category, variables in VARIABLE_CATEGORIES.items()
    )

    # radius over time
    radius_over_time = (
        pn.ggplot(
            rhos.with_columns(
                # demean
                pl.col("rho").pipe(lambda x: (x / x.mean()).over("category"))
            ),
            pn.aes("year", "rho"),
        )
        + pn.geom_point(color="steelblue")
        + pn.facet_wrap(
            "category",
            scales="free",
            labeller=labeller(cols=lambda s: r"\scshape " + s),  # type: ignore
        )
        # covid
        + pn.geom_vline(
            xintercept=2020,
            linetype="dashed",
            alpha=0.8,
            color="orange",
        )
        # financial crisis
        + pn.geom_vline(
            xintercept=2007,
            linetype="dashed",
            alpha=0.8,
            color="green",
        )
        + theme_ali()
    )
    radius_over_time.save(
        "figures/polarization_by_category.pgf",
        width=11 - 3,
        height=8.5 - 3.5,  # leave room for caption
    )

    # proportion over time
    # (
    #     pn.ggplot(rhos, pn.aes("year", "trace", color="category"))
    #     + pn.geom_point()
    #     + pn.facet_wrap("category", scales="free_y")
    # ).draw(show=True)

    # (
    #     pn.ggplot(
    #         rhos.unpivot(["rho", "trace", "proportion"], index="year"),
    #         pn.aes("year", "value", color="variable"),
    #     )
    #     + pn.geom_point()
    #     + pn.facet_wrap("variable", scales="free_y")
    # ).draw(show=True)


if __name__ == "__main__":
    main()
