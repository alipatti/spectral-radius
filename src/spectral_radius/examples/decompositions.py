from typing import Literal

import numpy as np
import plotnine as pn

from spectral_radius.examples.helpers import (
    create_sample_df,
    create_ellipses_df,
    OUTPUT,
    R,
    MuSigmaList,
)
from spectral_radius.plot_helpers import BASE_ELLIPSE_PLOT


def _stretch(x: np.ndarray) -> float:
    return np.linalg.norm(x, 2) / x.trace()


def stretch_v_scale_decomp_plot(
    stretch_or_scale: Literal["stretched", "scaled"],
    s=1.4,
) -> pn.ggplot:
    original = np.diag([s, 1])

    if stretch_or_scale == "scaled":
        new = original * s
        assert np.allclose(_stretch(original), _stretch(new))
    elif stretch_or_scale == "stretched":
        new = original + np.diag([s**2 - s, -(s**2) + s])
        assert np.allclose(original.trace(), new.trace())
    else:
        raise ValueError

    origin = (0, 0)

    # rotate so they look better
    for m in original, new:
        m @= np.linalg.matrix_power(R, 2)

    mus_and_sigmas = (origin, original), (origin, new)

    w = (8.5 - 2) / 3
    p = (
        BASE_ELLIPSE_PLOT
        + pn.geom_point(
            data=create_sample_df(mus_and_sigmas),
            alpha=0.2,
            shape="x",
            size=1,
        )
        + pn.geom_path(
            data=create_ellipses_df(mus_and_sigmas),
            linetype="dashed",
            alpha=0.5,
        )
        + pn.theme(figure_size=(w * 2.5 / 2, w))
    )

    return p


def group_decomp_plot(mus_and_sigmas: MuSigmaList) -> pn.ggplot:
    mus = np.stack([mu for mu, _ in mus_and_sigmas])
    sigmas = np.stack([sigma for _, sigma in mus_and_sigmas])

    mu = mus.mean(axis=0)

    sigma_within = sigmas.mean(axis=0)
    sigma_between = (mus - mu).T @ (mus - mu) / len(mus)

    sigma = sigma_within + sigma_between

    sample_points = create_sample_df(mus_and_sigmas)

    within_ellipses = create_ellipses_df(mus_and_sigmas)
    # between_elipses = create_ellipses_df([([0, 0], sigma_between)])

    pooled_ellipse = create_ellipses_df([(mu, sigma)]).drop("sample")

    w = (8.5 - 2) / 3
    return (
        BASE_ELLIPSE_PLOT
        # + AXES
        + pn.geom_point(data=sample_points, alpha=0.2, shape="x", size=1)
        + pn.geom_point(data=within_ellipses.group_by("sample").mean(), shape="x")
        + pn.geom_point(data=pooled_ellipse.mean(), shape="x", color="black")
        + pn.geom_path(
            data=within_ellipses,
            linetype="dashed",
            alpha=0.5,
        )
        + pn.geom_path(
            color="black",
            data=pooled_ellipse,
            # linetype="dashed",
        )
        + pn.theme(figure_size=(w, w))
    )


def main():
    examples = {
        "negative": [
            ([0.2, 0.05], 1.5 * np.diag([1, 0.2]) @ R @ R),  # up/down
            ([-0.2, -0.05], 1.5 * np.diag([0.2, 1]) @ R),  # right/left
        ],
        "positive": [
            ([-0.3, -0.3], (1 - np.diag([0, 0.5])) @ R),
            ([0.4, 0.3], (1 - np.diag([0, 0.5]))),
        ],
        "zero": [
            (np.array([0.0, 0.5]) @ R, np.diag([1.5, 0.2]) @ R),
            (np.array([0.0, -0.5]) @ R, np.diag([1.5, 0.2]) @ R),
        ],
    }

    out = OUTPUT / "decomps" / "by_group"
    out.mkdir(exist_ok=True, parents=True)
    for name, mus_and_sigmas in examples.items():
        group_decomp_plot(mus_and_sigmas).save(out / f"{name}.pdf")

    out = OUTPUT / "decomps" / "stretch_v_scale"
    out.mkdir(exist_ok=True, parents=True)
    for name in "stretched", "scaled":
        stretch_v_scale_decomp_plot(name).save(out / f"{name}.pdf")
