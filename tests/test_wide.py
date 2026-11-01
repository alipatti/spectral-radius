import polars as pl
import pytest
from helpers import survey_data
from polars.testing import assert_series_equal

from spectral_radius.surveys import DEMOGRAPHICS, SurveyData, wide


def respondent(survey: str, year: int, id: int, weight: float = 1.0) -> dict:
    return {"survey": survey, "year": year, "respondent": id, "weight": weight}


def answer(
    survey: str,
    year: int,
    id: int,
    variable: str,
    value: int | None,
) -> dict:
    return {
        "survey": survey,
        "year": year,
        "respondent": id,
        "variable": variable,
        "value": value,
    }


@pytest.fixture
def data() -> SurveyData:
    return survey_data(
        [
            respondent("a", 1990, 1, weight=2.0),
            respondent("a", 2000, 1),
            respondent("a", 2000, 2),
            respondent("a", 2010, 1),
            respondent("b", 2000, 1),
        ],
        [
            answer("a", 1990, 1, "q", 0),
            answer("a", 2000, 1, "q", 1),
            answer("a", 2000, 2, "q", 3),
            answer("a", 2000, 2, "r", 5),
            answer("a", 2010, 1, "q", 5),
            answer("b", 2000, 1, "q", 100),
        ],
    )


def test_columns(data: SurveyData):
    """Give year, demographics, one column per unique variable, and weight."""
    columns = wide(data, "a", ["q", "r", "q"]).columns

    assert columns == ["year", *DEMOGRAPHICS, "q", "r", "w"]


def test_keeps_every_respondent_of_survey(data: SurveyData):
    """Keep respondents with no answers, and only those of the survey."""
    out = wide(data, "a", ["r"], encoding=lambda x: x)

    assert out["year"].to_list() == [1990, 2000, 2000, 2010]
    assert out["r"].to_list() == [None, None, 5, None]
    assert out["w"].to_list() == [2.0, 1.0, 1.0, 1.0]


def test_encodes_over_observed_range_after_year_filter(data: SurveyData):
    """Scale onto [-1, 1] using the range observed from `min_year` onward."""
    out = wide(data, "a", ["q"], min_year=2000)

    assert_series_equal(out["q"], pl.Series("q", [-1.0, 0.0, 1.0]))


def test_unanswered_variable_is_null(data: SurveyData):
    """Give a null column for a variable no respondent answered."""
    out = wide(data, "a", ["unasked"], encoding=lambda x: x)

    assert out["unasked"].dtype == pl.Int64
    assert out["unasked"].null_count() == len(out)


def test_non_answers_are_null():
    """Null every missing kind, without it counting toward the encoded range."""
    data = survey_data(
        [respondent("a", 2000, i) for i in (1, 2, 3)],
        [
            answer("a", 2000, 1, "q", 0),
            answer("a", 2000, 2, "q", 4),
            answer("a", 2000, 3, "q", None) | {"missing": "dont_know"},
        ],
    )

    out = wide(data, "a", ["q"])

    assert out["q"].to_list() == [-1.0, 1.0, None]
