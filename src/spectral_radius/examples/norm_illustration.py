import numpy as np
import plotnine as pn

from spectral_radius.examples.helpers import OUTPUT, create_ellipses_df
from spectral_radius.plot_helpers import BASE_ELLIPSE_PLOT, savefig


def main():
    a, b = 3, 1
    padding = 0.4

    ellipses = create_ellipses_df([([0, 0], np.diag([a, b]))])

    p = (
        BASE_ELLIPSE_PLOT
        + pn.geom_path(data=ellipses, color="black")
        + pn.theme(figure_size=(5, 3))
        # + pn.theme_classic()
        + pn.lims(x=(-3.7, 3.1), y = (-1.7, 1.2))
        + pn.geom_segment(
            pn.aes(x=-a, xend=a, y=-b - padding, yend=-b - padding),
            inherit_aes=False,
            arrow=pn.arrow(ends="both", length=0.07, type="closed"),
        )
        + pn.geom_segment(
            pn.aes(y=-b, yend=b, x=-a - padding, xend=-a - padding),
            inherit_aes=False,
            arrow=pn.arrow(ends="both", length=0.07),
        )
        + pn.annotate("text", x=0, y=-b - padding * 1.5, label="$a$")
        + pn.annotate("text", y=0, x=-a - padding * 1.5, label="$b$", angle=90)
    )
    savefig(p, OUTPUT / "motivation" / "norm_illustration.pdf")
