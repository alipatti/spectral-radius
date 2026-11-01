"""Analysis-facing views of `SurveyData`."""

from collections.abc import Collection, Mapping, Sequence

import polars as pl

from spectral_radius.surveys.encodings import Encoding, scale_valid_range_to_pm_1
from spectral_radius.surveys.schemas import DEMOGRAPHICS, SurveyData, VariableScale


def declared_scales(
    data: SurveyData,
    survey: str,
    variables: Sequence[str],
) -> dict[str, VariableScale]:
    """The declared scale of each of `variables`, raising for any `survey` doesn't read."""
    rows = data.codebook.filter(
        pl.col("survey") == survey,
        pl.col("in_responses"),
        pl.col("variable").cast(pl.String).is_in(variables),
    ).select(pl.col("variable").cast(pl.String), "valid", "scale")
    scales = {
        variable: VariableScale(valid["min"], valid["max"], scale)
        for variable, valid, scale in rows.iter_rows()
    }

    if unread := [v for v in variables if v not in scales]:
        raise ValueError(f"Survey {survey!r} doesn't read variables {unread}")

    return scales


def _require_categorical_allowed(
    scales: Mapping[str, VariableScale],
    allow_categorical: Collection[str],
) -> None:
    categorical = {v for v, s in scales.items() if s.scale == "categorical"}

    if disallowed := sorted(categorical.difference(allow_categorical)):
        raise ValueError(
            f"Variables {disallowed} have categorical scales, "
            "which encodings treat as ordinal; "
            "pass them in `allow_categorical` to accept that"
        )

    if stale := sorted(set(allow_categorical).difference(categorical)):
        raise ValueError(f"Variables {stale} are allowed as categorical but aren't")


def wide(
    data: SurveyData,
    survey: str,
    variables: Sequence[str],
    *,
    encoding: Encoding = scale_valid_range_to_pm_1,
    min_year: int | None = None,
    allow_categorical: Collection[str] = (),
) -> pl.DataFrame:
    """
    One row per respondent with one column per variable, plus demographics and weight `w`.

    Rows are respondents from `min_year` onward (all years if `None`).
    Every non-answer is null, whatever its `missing` kind.
    `encoding` (see `encodings`) is applied to each variable over those rows,
    given the variable's declared scale;
    with the default, it scales each variable's valid range onto [-1, 1].
    Every variable must be one the survey reads,
    and those with categorical scales must be named in `allow_categorical`,
    since encodings treat codes as ordered.
    """
    variables = list(dict.fromkeys(variables))
    scales = declared_scales(data, survey, variables)
    _require_categorical_allowed(scales, allow_categorical)

    in_survey = pl.col("survey") == survey
    in_years = pl.lit(True) if min_year is None else pl.col("year") >= min_year

    respondents = data.respondents.filter(in_survey, in_years)
    # a non-answer's `value` is null, so pivoting `value` alone nulls every kind
    answers = (
        data.responses.filter(in_survey, in_years, pl.col("variable").is_in(variables))
        .select("year", "respondent", pl.col("variable").cast(pl.String), "value")
        .pivot(on="variable", index=["year", "respondent"], values="value")
    )

    # named `w` before encoding, since some encodings weight by it
    return (
        respondents.join(
            answers,
            on=["year", "respondent"],
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        .rename({"weight": "w"})
        .select(
            "year",
            *DEMOGRAPHICS,
            *(pl.col(v).pipe(encoding, scales[v]) for v in variables),
            "w",
        )
    )
