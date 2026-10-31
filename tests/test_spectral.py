from collections.abc import Collection, Iterator, Mapping
from itertools import combinations_with_replacement
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spectral_radius import spectral
from spectral_radius.constants import START_YEAR
from spectral_radius.spectral import (
    binned_year,
    measures_by_group_and_category,
    pairwise_complete_covariance,
    polarization_over_time_data,
    spectral_measures,
)

N_ROWS = 400
CATEGORIES = {
    "partial_and_disjoint": ["dense", "sparse", "even_rows", "odd_rows"],
    "with_null_in_a_group": ["dense", "null_in_year_3"],
}


@pytest.fixture
def survey() -> pl.DataFrame:
    """Synthetic survey whose columns have different missingness patterns."""
    rng = np.random.default_rng(0)
    row = np.arange(N_ROWS)
    year = rng.integers(1, 4, N_ROWS)

    def with_nulls(missing: np.ndarray) -> pl.Series:
        values = np.where(missing, np.nan, rng.normal(size=N_ROWS))
        return pl.Series(values).fill_nan(None)

    return pl.DataFrame(
        {
            "year": year,
            "g": pl.Series(rng.choice(["a", "b"], N_ROWS)).cast(pl.Categorical),
            "w": rng.uniform(0.5, 2, N_ROWS),
            # partial overlap with each other
            "dense": with_nulls(rng.random(N_ROWS) < 0.2),
            "sparse": with_nulls(rng.random(N_ROWS) < 0.6),
            # never observed together
            "even_rows": with_nulls(row % 2 == 1),
            "odd_rows": with_nulls(row % 2 == 0),
            # entirely missing in one group
            "null_in_year_3": with_nulls((year == 3) | (rng.random(N_ROWS) < 0.1)),
        }
    )


def reference_covariance(
    x: np.ndarray,
    observed: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Explicit per-pair loop over the rows where both variables are observed."""
    n = x.shape[1]
    sigma = np.zeros((n, n))

    for a, b in combinations_with_replacement(range(n), 2):
        both = observed[:, a] & observed[:, b]
        total_weight = w[both].sum()

        if total_weight > 0:
            numerator = (w * x[:, a] * x[:, b])[both].sum()
            sigma[a, b] = sigma[b, a] = numerator / total_weight

    return sigma


def reference_measures(
    df: pl.DataFrame,
    categories: Mapping[str, Collection[str]],
    group: Collection[str],
) -> pl.DataFrame:
    """The original per-pair polars implementation, one group at a time."""

    def measures(frame: pl.DataFrame, columns: Collection[str]) -> dict:
        frame = frame.with_columns(pl.col(columns).pipe(lambda x: x - x.mean()))
        n = len(columns)
        sigma = np.empty((n, n))
        weights = frame["w"]

        for (i, a), (j, b) in combinations_with_replacement(enumerate(columns), 2):
            product = frame[b] * frame[a] * weights
            total_weight = weights.filter(product.is_not_null()).sum()
            sigma[i, j] = sigma[j, i] = (
                float(product.sum()) / float(total_weight) if total_weight > 0 else 0
            )

        return spectral_measures(sigma) | {"w": frame["w"].sum()}

    return pl.DataFrame(
        {"category": cat}
        | dict(zip(group, keys, strict=True))
        | measures(frame, columns)
        for cat, columns in categories.items()
        for keys, frame in df.drop_nulls(group).group_by(group)
    ).sort("*")


def test_covariance_matches_per_pair_loop(survey: pl.DataFrame):
    """Match an explicit per-pair loop, with weights and every missingness pattern."""
    columns = list(
        dict.fromkeys(CATEGORIES["partial_and_disjoint"] + ["null_in_year_3"])
    )
    in_year_3 = survey.filter(pl.col("year") == 3)
    demeaned = in_year_3.select(pl.col(columns) - pl.col(columns).mean())

    x = demeaned.fill_null(0).to_numpy()
    observed = demeaned.select(pl.all().is_not_null()).to_numpy()
    w = in_year_3["w"].to_numpy()

    np.testing.assert_allclose(
        pairwise_complete_covariance(x, observed, w),
        reference_covariance(x, observed, w),
        rtol=1e-12,
        atol=1e-15,
    )


def test_covariance_is_zero_without_overlap(survey: pl.DataFrame):
    """Give 0 for pairs never observed together and for a fully missing column."""
    in_year_3 = survey.filter(pl.col("year") == 3)
    columns = ["even_rows", "odd_rows", "null_in_year_3"]
    values = in_year_3.select(columns)

    sigma = pairwise_complete_covariance(
        values.fill_null(0).to_numpy(),
        values.select(pl.all().is_not_null()).to_numpy(),
        in_year_3["w"].to_numpy(),
    )

    assert sigma[0, 1] == sigma[1, 0] == 0
    assert (sigma[2] == 0).all()
    assert sigma[0, 0] > 0
    assert sigma[1, 1] > 0


@pytest.mark.parametrize("group", [["year"], ["year", "g"]])
def test_measures_match_original_implementation(survey: pl.DataFrame, group: list[str]):
    """Reproduce the original per-pair implementation's measures for every group."""
    assert_frame_equal(
        measures_by_group_and_category(survey, CATEGORIES, group),
        reference_measures(survey, CATEGORIES, group),
        check_exact=False,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    ("bin_size", "year", "expected"),
    [
        (5, START_YEAR, START_YEAR + 2.5),
        (5, START_YEAR + 4, START_YEAR + 2.5),
        (5, START_YEAR + 5, START_YEAR + 7.5),
        (5, START_YEAR - 1, START_YEAR - 2.5),
        (1, START_YEAR + 3, START_YEAR + 3.5),
    ],
)
def test_binned_year_maps_to_bin_midpoint(bin_size: int, year: int, expected: float):
    """Map each year to the midpoint of its bin, with bins starting at START_YEAR."""
    years = pl.DataFrame({"year": [year]})
    assert years.select(binned_year(bin_size)).item() == expected


def test_binned_year_none_leaves_year_unchanged():
    """Return the year itself, with its dtype, when there is no bin size."""
    years = pl.DataFrame({"year": [1990, 1997, 2021]}, schema={"year": pl.Int16})
    assert_frame_equal(years.select(binned_year(None)), years)


@pytest.fixture
def measure_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[list[str]]]:
    """Serve a synthetic survey and record the groups of each measures computation."""
    rng = np.random.default_rng(1)
    n = 200
    responses = pl.DataFrame(
        {
            "year": rng.integers(START_YEAR, START_YEAR + 10, n),
            "race": rng.choice(["a", "b"], n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "w": rng.uniform(0.5, 2, n),
        }
    )
    calls: list[list[str]] = []

    def recording_measures(*args, **kwargs):
        calls.append(list(args[2]))
        return measures_by_group_and_category(*args, **kwargs)

    monkeypatch.setitem(spectral.QUESTION_SETS, "fake", {"xy": ["x", "y"]})
    monkeypatch.setattr(spectral, "analysis_frame", lambda survey: responses)
    monkeypatch.setattr(
        spectral,
        "survey_data",
        lambda: SimpleNamespace(require_demographic=lambda survey, by: None),
    )
    monkeypatch.setattr(spectral, "measures_by_group_and_category", recording_measures)

    for cached in spectral._pooled_measures, spectral._within_measures:
        cached.cache_clear()

    yield calls

    for cached in spectral._pooled_measures, spectral._within_measures:
        cached.cache_clear()


def test_polarization_over_time_data_computes_pooled_measures_once(
    measure_calls: list[list[str]],
):
    """Compute the pooled measures once across equivalently passed arguments."""
    polarization_over_time_data(survey="fake")
    polarization_over_time_data(None, year_bin_width=5, survey="fake")
    polarization_over_time_data("race", survey="fake")
    polarization_over_time_data("race", metric="trace", survey="fake")

    assert measure_calls == [["year"], ["year", "race"]]
