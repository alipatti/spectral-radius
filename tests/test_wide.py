from collections.abc import Sequence

import polars as pl
import pytest
from helpers import survey_data, unlabeled_scale
from polars.testing import assert_series_equal

from spectral_radius.surveys import DEMOGRAPHICS, SurveyData, wide
from spectral_radius.surveys.encodings import scale_observed_range_to_pm_1


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


def identity(x: pl.Expr, _codes: Sequence[int]) -> pl.Expr:
    return x


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
        scales={
            ("a", "q"): unlabeled_scale("zero_to_ten", range(11)),
            ("a", "r"): unlabeled_scale("shuffled", [1, 3, 2, 4, 5]),
        },
    )


def test_columns(data: SurveyData):
    """Give year, demographics, one column per unique variable, and weight."""
    columns = wide(data, "a", ["q", "r", "q"], allow_categorical=["r"]).columns

    assert columns == ["year", *DEMOGRAPHICS, "q", "r", "w"]


def test_keeps_every_respondent_of_survey(data: SurveyData):
    """Keep respondents with no answers, and only those of the survey."""
    out = wide(data, "a", ["r"], encoding=identity, allow_categorical=["r"])

    assert out["year"].to_list() == [1990, 2000, 2000, 2010]
    assert out["r"].to_list() == [None, None, 5, None]
    assert out["w"].to_list() == [2.0, 1.0, 1.0, 1.0]


def test_encodes_code_range_by_default(data: SurveyData):
    """Scale the range of the scale's codes onto [-1, 1], whatever is observed."""
    out = wide(data, "a", ["q"], min_year=2000)

    assert_series_equal(out["q"], pl.Series("q", [-0.8, -0.4, 0.0]))


def test_observed_range_encoding_uses_rows_after_year_filter(data: SurveyData):
    """Scale onto [-1, 1] the range observed from `min_year` onward."""
    out = wide(data, "a", ["q"], min_year=2000, encoding=scale_observed_range_to_pm_1)

    assert_series_equal(out["q"], pl.Series("q", [-1.0, 0.0, 1.0]))


def test_rejects_variable_survey_doesnt_read(data: SurveyData):
    """Refuse a variable the survey doesn't read, rather than giving it nulls."""
    with pytest.raises(ValueError, match=r"'b' doesn't read variables \['r'\]"):
        wide(data, "b", ["q", "r"])


def test_rejects_categorical_variable_not_allowed(data: SurveyData):
    """Refuse a variable with codes out of scale order unless explicitly allowed."""
    with pytest.raises(ValueError, match=r"\['r'\] have scale codes out of"):
        wide(data, "a", ["q", "r"])


def test_rejects_allowing_non_categorical_variable(data: SurveyData):
    """Refuse an allowance for a variable whose codes are in scale order."""
    with pytest.raises(ValueError, match=r"\['q'\] are allowed as categorical"):
        wide(data, "a", ["q", "r"], allow_categorical=["q", "r"])


def test_non_answers_are_null():
    """Null every missing kind."""
    data = survey_data(
        [respondent("a", 2000, i) for i in (1, 2, 3)],
        [
            answer("a", 2000, 1, "q", 0),
            answer("a", 2000, 2, "q", 4),
            answer("a", 2000, 3, "q", None) | {"missing": "dont_know"},
        ],
        scales={("a", "q"): unlabeled_scale("zero_to_four", range(5))},
    )

    out = wide(data, "a", ["q"])

    assert out["q"].to_list() == [-1.0, 1.0, None]
