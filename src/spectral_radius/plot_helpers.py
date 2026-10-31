from pathlib import Path
from typing import Any

import matplotlib as mpl
import plotnine as pn
import polars as pl
from matplotlib.typing import RcKeyType

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
    ncol=2,
)

COLOR_SCALE = pn.scale_color_manual(COLORS)
PERCENT_CHANGE_SCALE = pn.scale_y_continuous(
    name="Percent Change",
    labels=lambda ss: [f"{s - 1:+.0%}" for s in ss],  # type: ignore
)

BASE_ELLIPSE_PLOT = (
    pn.ggplot(pl.DataFrame(), pn.aes("x1", "x2", color="factor(sample)"))
    + pn.coord_fixed()
    + pn.theme_void(base_size=11)
    + pn.guides(fill=False, color=False)
    + pn.lims(x=(-2.5, 2.5), y=(-2, 2))
    + COLOR_SCALE
)


PGF_PARAMS: dict[RcKeyType, Any] = {
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

        print(f"Saving figure to {path}")

        with mpl.rc_context(PGF_PARAMS):
            fig.save(path, verbose=False, backend="pgf")
