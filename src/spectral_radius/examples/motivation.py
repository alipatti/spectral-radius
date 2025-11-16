from plotnine_theme import margin, theme_ali
import polars as pl
import plotnine as pn
from polars_utils.covariance import covariance_matrix
from polars_utils.stats import mean
import numpy as np

from spectral_radius.gss import load_gss_data
from spectral_radius.plot_helpers import COLORS, savefig
from spectral_radius.variable_coding import scale_to_pm_1
from spectral_radius.examples.helpers import OUTPUT, create_ellipses_df


def gss_norm_example():
    gss = load_gss_data()[0]

    x, y = "helpsick", "helppoor"
    years = 2000, 2012, 2024

    df = gss.select(
        "year",
        pl.col(x, y).pipe(scale_to_pm_1).neg(),
        pl.col("wtssps").alias("w"),
    ).filter(pl.col("year").is_in(years))

    mus_and_sigmas = {
        ks[0]: (
            g.select(pl.col(x, y).pipe(mean, w="w")).to_numpy().flatten(),
            covariance_matrix(g, w="w", columns=[x, y]),
        )
        for ks, g in df.partition_by("year", as_dict=True).items()
    }

    rhos = pl.DataFrame(
        dict(
            year=year,
            rho=np.linalg.norm(sigma, 2).__float__(),
            trace=np.linalg.norm(sigma, "nuc").__float__(),
            frob=np.linalg.norm(sigma, "fro").__float__(),
        )
        for year, (_, sigma) in mus_and_sigmas.items()
    ).with_columns(
        label=pl.format(
            r"""
            \footnotesize $||\Sigma||_2 = {}$
            \footnotesize $||\Sigma||_F = {}$
            \footnotesize $||\Sigma||_* = {}$""",
            *(
                pl.col(c).round(2).cast(str).str.pad_end(4, "0")
                for c in ("rho", "frob", "trace")
            ),
        ),
    )

    p = (
        pn.ggplot(
            df.drop_nulls()
            # stratified sample
            .group_by("year", maintain_order=True)
            .map_groups(lambda g: g.sample(100, seed=1280))
            .select("year", x1=x, x2=y),
            pn.aes("x1", "x2"),
        )
        + pn.geom_point(
            alpha=0.15,
            shape="x",
            size=1,
            position=pn.position_jitter(0.1, 0.1),
            color=COLORS[0],
        )
        + pn.geom_path(
            data=create_ellipses_df(mus_and_sigmas).rename(dict(sample="year")),
            # linetype="dashed",
            # alpha=0.5,
        )
        + pn.geom_text(
            pn.aes(x=-0.5, y=+0.75, label="label"),
            data=rhos,
            # format_string="\\small $\\rho={:.2f}$",
            inherit_aes=False,
        )
        + pn.coord_fixed()
        + pn.theme_classic(base_size=10)
        + pn.theme(
            plot_title=pn.element_text(size=10),
            strip_text=pn.element_text(
                ha="center",
                margin=margin(10, 0, 0, 0),
            ),
            strip_background=pn.element_blank(),
            axis_text=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            panel_spacing=0.05,
            figure_size=(8.5 - 2, 3),
            text=pn.element_text(family="cm"),
        )
        + pn.facet_wrap("year")
        + pn.labs(
            title="Degree to which the government should help...",
            x="...the sick",
            y="...the poor",
        )
    )

    savefig(p, OUTPUT / "motivation" / "example", size=(8.5 - 2, 3))


def main():
    gss_norm_example()
