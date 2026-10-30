"""Analysis-facing views of `SurveyData`."""

from collections.abc import Callable, Sequence

import polars as pl

from spectral_radius.surveys.encodings import scale_to_pm_1
from spectral_radius.surveys.schemas import DEMOGRAPHICS, SurveyData

type Encoding = Callable[[pl.Expr], pl.Expr]


def wide(
    data: SurveyData,
    survey: str,
    variables: Sequence[str],
    *,
    encoding: Encoding = scale_to_pm_1,
    min_year: int | None = None,
) -> pl.DataFrame:
    """
    One row per respondent with one column per variable, plus demographics and weight `w`.

    Rows are respondents from `min_year` onward (all years if `None`).
    `encoding` (see `encodings`) is applied to each variable over those rows,
    so with the default it scales each variable's observed range onto [-1, 1].
    """
    # TODO: scaling by the codebook's range of valid values,
    # rather than the observed range, would be more principled
    variables = list(dict.fromkeys(variables))
    in_survey = pl.col("survey") == survey
    in_years = pl.lit(True) if min_year is None else pl.col("year") >= min_year

    respondents = data.respondents.filter(in_survey, in_years)
    answers = (
        data.responses.filter(in_survey, in_years, pl.col("variable").is_in(variables))
        .with_columns(pl.col("variable").cast(pl.String))
        .pivot(on="variable", index=["year", "respondent"], values="value")
    )
    # variables nobody answered get no column from the pivot
    unanswered = (
        pl.lit(None, dtype=pl.Int64).alias(v)
        for v in variables
        if v not in answers.columns
    )

    # named `w` before encoding, since some encodings weight by it
    return (
        respondents.join(
            answers.with_columns(*unanswered),
            on=["year", "respondent"],
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        .rename({"weight": "w"})
        .select("year", *DEMOGRAPHICS, pl.col(variables).pipe(encoding), "w")
    )
