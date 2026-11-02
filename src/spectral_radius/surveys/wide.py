"""Analysis-facing views of `SurveyData`."""

from collections.abc import Collection, Mapping, Sequence

import polars as pl

from spectral_radius.surveys.encodings import Encoding, scale_code_range_to_pm_1
from spectral_radius.surveys.schemas import DEMOGRAPHICS, SurveyData


def declared_scale_codes(
    data: SurveyData,
    survey: str,
    variables: Sequence[str],
) -> dict[str, list[int]]:
    """The scale codes of each of `variables`, raising for any `survey` doesn't read."""
    rows = data.codebook.filter(
        pl.col("survey") == survey,
        pl.col("in_responses"),
        pl.col("variable").cast(pl.String).is_in(variables),
    ).select(pl.col("variable").cast(pl.String), "scale_codes")
    codes = dict(rows.iter_rows())

    if unread := [v for v in variables if v not in codes]:
        raise ValueError(f"Survey {survey!r} doesn't read variables {unread}")

    return codes


def _require_categorical_allowed(
    codes: Mapping[str, Sequence[int]],
    allow_categorical: Collection[str],
) -> None:
    categorical = {v for v, c in codes.items() if c != sorted(c)}

    if disallowed := sorted(categorical.difference(allow_categorical)):
        raise ValueError(
            f"Variables {disallowed} have scale codes out of scale order, "
            "which encodings treat as ordered by code; "
            "pass them in `allow_categorical` to accept that"
        )

    if stale := sorted(set(allow_categorical).difference(categorical)):
        raise ValueError(
            f"Variables {stale} are allowed as categorical but their codes are ordered"
        )


def wide(
    data: SurveyData,
    survey: str,
    variables: Sequence[str],
    *,
    encoding: Encoding = scale_code_range_to_pm_1,
    min_year: int | None = None,
    allow_categorical: Collection[str] = (),
) -> pl.DataFrame:
    """
    One row per respondent with one column per variable, plus demographics and weight `w`.

    Rows are respondents from `min_year` onward (all years if `None`).
    Every non-answer is null, whatever its `missing` kind.
    `encoding` (see `encodings`) is applied to each variable over those rows,
    given the variable's scale codes;
    with the default, it scales the range of the codes onto [-1, 1].
    Every variable must be one the survey reads,
    and those whose scale codes are out of scale order
    must be named in `allow_categorical`, since encodings treat codes as ordered.
    """
    variables = list(dict.fromkeys(variables))
    codes = declared_scale_codes(data, survey, variables)
    _require_categorical_allowed(codes, allow_categorical)

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
            *(pl.col(v).pipe(encoding, codes[v]) for v in variables),
            "w",
        )
    )
