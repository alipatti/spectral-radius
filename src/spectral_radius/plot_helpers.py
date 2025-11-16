from pathlib import Path

import matplotlib as mpl
import polars as pl
import plotnine as pn

COLORS = [
    "#4976ca",  # blue
    "#64a536",  # green
    "#ee6449",  # red
    "#ffa600",  # yellow
    "#ffee88",
    "#7b506f",
]

CATEGORY_WRAP = pn.facet_wrap(
    "category",
    scales="free",
    labeller=pn.labeller(cols=lambda s: rf"\scshape {s}"),  # type: ignore
)

COLOR_SCALE = pn.scale_color_manual(COLORS)
PERCENT_CHANGE_SCALE = pn.scale_y_continuous(
    name="Percent Change",
    labels=lambda ss: [f"{s - 1:+.0%}" for s in ss],  # type: ignore
)

COVID_VLINE = pn.geom_vline(
    xintercept=2020.25,
    linetype="dashed",
    alpha=0.5,
    color="black",
)

RECESSION_VLINE = pn.geom_vline(
    xintercept=2007.25,
    linetype="dashed",
    alpha=0.5,
    color="black",
)


BASE_ELLIPSE_PLOT = (
    pn.ggplot(pl.DataFrame(), pn.aes("x1", "x2", color="factor(sample)"))
    + pn.coord_fixed()
    + pn.theme_void(base_size=11)
    + pn.guides(fill=False, color=False)
    + pn.lims(x=(-2.5, 2.5), y=(-2, 2))
    + COLOR_SCALE
)


PGF_PARAMS = {
    "pgf.texsystem": "pdflatex",
    "font.family": "cmu serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": "\\usepackage{lmodern}\n\\usepackage[T1]{fontenc}",
}


def savefig(
    fig: pn.ggplot,
    path: Path,
    size: None | tuple[float | int, float | int] = (11 - 2, 8.5 - 3),
    extensions=(".pdf", ".pgf"),
) -> None:
    if size is not None:
        fig = fig + pn.theme(figure_size=size)

    for extension in extensions:
        path = path.with_suffix(extension)
        path.parent.mkdir(exist_ok=True, parents=True)

        # get current settings
        backend, params = mpl.get_backend(), mpl.rcParams.copy()

        print(f"Saving figure to {path}")

        try:
            # use pgf backend to save
            mpl.use("pgf")
            mpl.rcParams.update(PGF_PARAMS)
            fig.save(path, verbose=False)

        finally:
            # restore settings
            mpl.use(backend)
            mpl.rcParams.update(params)
