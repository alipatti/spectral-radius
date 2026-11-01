"""
Check the analysis frame built from the real cached GSS data.

These fail (rather than skip) when the data is missing;
deselect them with `-m 'not data'`.
"""

import contextlib
from pathlib import Path

import polars as pl
import pytest

from spectral_radius.analysis import analysis_frame, survey_data
from spectral_radius.constants import START_YEAR
from spectral_radius.surveys import DEMOGRAPHICS, GSS, question_variables

pytestmark = pytest.mark.data

REPO = Path(__file__).parents[1]
VARIABLES = question_variables("gss")


@pytest.fixture(scope="module")
def responses() -> pl.DataFrame:
    assert GSS(root=REPO / "data").raw_dir.exists(), "Missing raw GSS data"

    with contextlib.chdir(REPO):
        return survey_data().responses.filter(pl.col("survey") == "gss")


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    # check first so a missing cache fails rather than starting a download
    assert GSS(root=REPO / "data").raw_dir.exists(), "Missing raw GSS data"

    # the pipeline reads from a path relative to the working directory
    with contextlib.chdir(REPO):
        return analysis_frame("gss")


def test_analysis_frame_columns(frame: pl.DataFrame):
    """List the year, demographics, question-set variables, and weight."""
    assert frame.columns == ["year", *DEMOGRAPHICS, *VARIABLES, "w"]


def test_analysis_frame_starts_at_start_year(frame: pl.DataFrame):
    """Keep only respondents from the start year on."""
    assert frame["year"].min() == START_YEAR


def test_analysis_frame_scales_variables_to_pm_1(frame: pl.DataFrame):
    """Scale every variable's observed range onto [-1, 1]."""
    variables = frame.select(VARIABLES)

    assert variables.min().row(0) == (-1.0,) * len(VARIABLES)
    assert variables.max().row(0) == (1.0,) * len(VARIABLES)


def test_responses_cover_every_respondent_and_variable(responses: pl.DataFrame):
    """Keep a row for every respondent and question-set variable."""
    n_respondents = responses.select("year", "respondent").n_unique()

    assert len(responses) == n_respondents * len(VARIABLES)


def test_responses_record_dont_knows_and_inapplicables(responses: pl.DataFrame):
    """Record "don't know" and "inapplicable" as kinds, not as dropped rows."""
    kinds = set(responses["missing"].drop_nulls().cast(pl.String))

    assert {"dont_know", "inapplicable"} <= kinds
