"""The data every analysis shares, loaded once per process."""

from functools import cache

import polars as pl

from spectral_radius.surveys import SurveyData, load, question_variables, wide
from spectral_radius.surveys.pipeline import source
from spectral_radius.surveys.question_sets import ALLOWED_CATEGORICAL


@cache
def survey_data() -> SurveyData:
    return load()


def start_year(survey: str) -> int:
    """The year from which the analysis uses respondents of `survey`."""
    return source(survey).start_year


@cache
def analysis_frame(survey: str) -> pl.DataFrame:
    """
    One row per respondent of `survey` from its start year on.

    Columns are the year, demographics, every variable in the survey's question
    sets (with its valid range scaled onto [-1, 1]), and the weight `w`.
    """
    return wide(
        survey_data(),
        survey,
        question_variables(survey),
        min_year=start_year(survey),
        allow_categorical=ALLOWED_CATEGORICAL[survey],
    )
