from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy._typing import _ArrayLikeFloat_co

from spectral_radius.constants import FIGURES

type MuAndSigma = tuple[_ArrayLikeFloat_co, _ArrayLikeFloat_co]
type MuSigmaList = Sequence[MuAndSigma] | Mapping[str, MuAndSigma]

OUTPUT = FIGURES / "examples/"

RNG = np.random.default_rng(seed=1281)

# small rotation matrix
SMALL_ANGLE = np.deg2rad(10)
R = np.array(
    [
        [np.cos(SMALL_ANGLE), -np.sin(SMALL_ANGLE)],
        [np.sin(SMALL_ANGLE), np.cos(SMALL_ANGLE)],
    ]
)


def df_from_numpy(arr: np.ndarray, c: Any) -> pl.DataFrame:
    return pl.from_numpy(arr, schema=["x1", "x2"]).with_columns(sample=pl.lit(c))


def ellipse_points(mu: _ArrayLikeFloat_co, sigma: _ArrayLikeFloat_co, n_points=100):
    t = np.linspace(0, 2 * np.pi, num=n_points)
    circle_points = np.column_stack((np.cos(t), np.sin(t)))
    return circle_points @ sigma + mu


def create_sample_df(mus_and_sigmas: MuSigmaList, n_samples=120):
    labels_mus_and_sigmas = (
        mus_and_sigmas.items()
        if isinstance(mus_and_sigmas, Mapping)
        else enumerate(mus_and_sigmas)
    )

    return pl.concat(
        df_from_numpy(RNG.multivariate_normal(mu, sigma, size=n_samples), c)
        for c, (mu, sigma) in labels_mus_and_sigmas
    )


def create_ellipses_df(mus_and_sigmas: MuSigmaList):
    labels_mus_and_sigmas = (
        mus_and_sigmas.items()
        if isinstance(mus_and_sigmas, Mapping)
        else enumerate(mus_and_sigmas)
    )

    return pl.concat(
        df_from_numpy(ellipse_points(mu, sigma), c)
        for c, (mu, sigma) in labels_mus_and_sigmas
    )
