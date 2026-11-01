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
from spectral_radius.surveys import DEMOGRAPHICS, GSS, SurveyData, question_variables
from spectral_radius.surveys.gss import OFF_SCALE_CODES, OPINION_SCALES

pytestmark = pytest.mark.data

REPO = Path(__file__).parents[1]
VARIABLES = question_variables("gss")


@pytest.fixture(scope="module")
def data() -> SurveyData:
    assert GSS(root=REPO / "data").raw_dir.exists(), "Missing raw GSS data"

    with contextlib.chdir(REPO):
        return survey_data()


@pytest.fixture(scope="module")
def responses(data: SurveyData) -> pl.DataFrame:
    return data.responses.filter(pl.col("survey") == "gss")


@pytest.fixture(scope="module")
def labels(data: SurveyData) -> dict[str, list[int]]:
    """The labeled values of each declared variable."""
    rows = data.codebook.filter(
        pl.col("survey") == "gss",
        pl.col("variable").cast(pl.String).is_in(list(OPINION_SCALES)),
    ).select(pl.col("variable").cast(pl.String), pl.col("labels").struct.field("value"))

    return dict(rows.iter_rows())


def test_declared_ranges_end_at_labeled_values(labels: dict[str, list[int]]):
    """Declare ranges whose ends the gss labels."""
    unlabeled = {
        (variable, end)
        for variable, scale in OPINION_SCALES.items()
        for end in (scale.min, scale.max)
        if end not in labels[variable]
    }

    assert not unlabeled


def test_labels_outside_declared_ranges_are_off_scale(labels: dict[str, list[int]]):
    """Declare ranges covering every labeled value but the declared off-scale ones."""
    outside = {
        (variable, value)
        for variable, scale in OPINION_SCALES.items()
        for value in labels[variable]
        if not scale.min <= value <= scale.max
    }
    off_scale = {
        (variable, code)
        for variable, codes in OFF_SCALE_CODES.items()
        for code in codes
    }

    assert outside == off_scale


def observed_ranges(responses: pl.DataFrame, min_year: int) -> dict[str, tuple]:
    rows = (
        responses.filter(pl.col("year") >= min_year)
        .group_by(pl.col("variable").cast(pl.String))
        .agg(pl.col("value").min().alias("low"), pl.col("value").max().alias("high"))
    )
    return {variable: (low, high) for variable, low, high in rows.iter_rows()}


@pytest.mark.parametrize("min_year", [0, START_YEAR], ids=["all_years", "analysis"])
def test_observed_ranges_span_declared_ranges(responses: pl.DataFrame, min_year: int):
    """
    Observe each variable's whole declared range, and nothing outside it.

    A narrower observed range would change the analysis
    from when it scaled the observed range;
    this lists any such variable.
    """
    observed = observed_ranges(responses, min_year)
    differing = {
        variable: {"observed": observed[variable], "declared": scale[:2]}
        for variable, scale in OPINION_SCALES.items()
        if variable in observed and observed[variable] != scale[:2]
    }

    assert not differing


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
    """Scale every variable's valid range, which each spans, onto [-1, 1]."""
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
