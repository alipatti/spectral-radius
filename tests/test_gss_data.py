"""
Check the new pipeline against the old `get_gss` on the real cached GSS data.

These fail (rather than skip) when the data is missing;
deselect them with `-m 'not data'`.
"""

import contextlib
from pathlib import Path

import polars as pl
import polars.selectors as cs
import pytest
from polars.testing import assert_frame_equal

from spectral_radius.constants import START_YEAR
from spectral_radius.surveys import GSS, SurveyData, load, question_variables, wide

pytestmark = pytest.mark.data

REPO = Path(__file__).parents[1]
OLD_CACHE = REPO / "data" / "gss"


@pytest.fixture(scope="module")
def data() -> SurveyData:
    gss = GSS(root=REPO / "data")
    # check first so a missing cache fails rather than starting a download
    assert gss.raw_dir.exists(), f"Missing raw GSS data at {gss.raw_dir}"

    return load([gss])


@pytest.fixture(scope="module")
def old() -> pl.DataFrame:
    assert OLD_CACHE.exists(), f"Missing old GSS cache at {OLD_CACHE}"

    from spectral_radius.gss import get_gss

    # the old loader reads from a path relative to the working directory
    with contextlib.chdir(REPO):
        return get_gss()


@pytest.fixture(scope="module")
def new(data: SurveyData) -> pl.DataFrame:
    return wide(data, "gss", question_variables("gss"), min_year=START_YEAR)


def as_strings(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns((cs.categorical() | cs.enum()).cast(pl.String))


def test_wide_matches_old_get_gss(new: pl.DataFrame, old: pl.DataFrame):
    """Reproduce the old analysis frame exactly, up to enum versus categorical."""
    assert_frame_equal(as_strings(new), as_strings(old))


def test_demographics_are_enums_where_old_were_categorical(
    new: pl.DataFrame,
    old: pl.DataFrame,
):
    """Differ in dtype only by giving demographics fixed categories."""
    differing = {
        c: (new[c].dtype, old[c].dtype)
        for c in new.columns
        if new[c].dtype != old[c].dtype
    }

    assert all(
        isinstance(n, pl.Enum) and isinstance(o, pl.Categorical)
        for n, o in differing.values()
    )


@pytest.mark.parametrize("by", [None, "religion", "education"])
def test_polarization_matches_old(new: pl.DataFrame, old: pl.DataFrame, by: str | None):
    """Give the same polarization measures as the old frame."""
    from spectral_radius.spectral import binned_year, measures_by_group_and_category
    from spectral_radius.surveys.question_sets import QUESTION_SETS

    def measures(frame: pl.DataFrame) -> pl.DataFrame:
        group = ["year"] if by is None else ["year", by]
        return (
            frame.with_columns(binned_year(5))
            .pipe(measures_by_group_and_category, QUESTION_SETS["gss"], group)
            .pipe(as_strings)
            .sort(pl.all())
        )

    assert_frame_equal(measures(new), measures(old))
