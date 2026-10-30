"""The data every analysis shares, loaded once per process."""

from functools import cache

import polars as pl

from spectral_radius.constants import START_YEAR
from spectral_radius.surveys import SurveyData, load, question_variables, wide


@cache
def survey_data() -> SurveyData:
    return load()


@cache
def analysis_frame(survey: str = "gss") -> pl.DataFrame:
    """
    One row per respondent of `survey` from `START_YEAR` on.

    Columns are the year, demographics, every variable in the survey's question
    sets (scaled onto [-1, 1]), and the weight `w`.
    """
    return wide(
        survey_data(),
        survey,
        question_variables(survey),
        min_year=START_YEAR,
    )
