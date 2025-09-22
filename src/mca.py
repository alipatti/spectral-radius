from typing import Iterable
import numpy as np
import polars as pl


def mca(
    df: pl.DataFrame,
    *,
    drop: Iterable[str] = ["year", "sample"],
):
    # https://en.wikipedia.org/wiki/Multiple_correspondence_analysis#Details

    X = df.drop(*drop, strict=False).to_dummies().to_numpy()

    # normalize X
    Z: np.ndarray = X / X.sum()

    row_sums: np.ndarray = Z.sum(axis=1)
    column_sums: np.ndarray = Z.sum(axis=0)
    residuals = Z - np.outer(row_sums, column_sums)

    M = np.diag(1 / np.sqrt(row_sums)) @ residuals @ np.diag(1 / np.sqrt(column_sums))

    return np.linalg.svd(M)

