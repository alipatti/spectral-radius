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
from spectral_radius.surveys import DEMOGRAPHICS, GSS, SurveyData, question_variables
from spectral_radius.surveys.scales import require_labels_match

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
def codebook(data: SurveyData) -> pl.DataFrame:
    return data.codebook.filter(pl.col("survey") == "gss")


def test_every_variable_read_has_a_scale_matching_its_labels(codebook: pl.DataFrame):
    """Assign every variable read a scale whose labels the gss publishes."""
    scales = GSS().scales

    assert set(codebook.filter("in_responses")["variable"].cast(pl.String)) == set(
        VARIABLES
    )
    require_labels_match(codebook, scales)


def test_off_scale_only_where_scale_declares_off_scale_codes(
    responses: pl.DataFrame,
):
    """Read off-scale answers only for variables whose scale declares such codes."""
    off_scale = set(
        responses.filter(pl.col("missing") == "off_scale")["variable"].cast(pl.String)
    )
    declared = {v for v, s in GSS().scales.items() if s.off_scale}

    assert off_scale <= declared


def observed_codes(responses: pl.DataFrame, min_year: int) -> dict[str, set[int]]:
    rows = (
        responses.filter(pl.col("year") >= min_year, pl.col("value").is_not_null())
        .group_by(pl.col("variable").cast(pl.String))
        .agg(pl.col("value").unique())
    )
    return {variable: set(values) for variable, values in rows.iter_rows()}


@pytest.mark.parametrize("min_year", [0, GSS.start_year], ids=["all_years", "analysis"])
def test_observed_codes_are_scale_codes(responses: pl.DataFrame, min_year: int):
    """
    Observe every code of each variable's scale, and nothing else.

    An unobserved code would make the observed-range encoding
    differ from the default one; this lists any such variable.
    """
    observed = observed_codes(responses, min_year)
    differing = {
        variable: {"observed": observed[variable], "scale": scale.codes}
        for variable, scale in GSS().scales.items()
        if observed[variable] != set(scale.codes)
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
    assert frame["year"].min() == GSS.start_year


def test_analysis_frame_scales_variables_to_pm_1(frame: pl.DataFrame):
    """Scale every variable onto [-1, 1], with every code observed."""
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
