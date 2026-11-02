"""Analysis-facing views of `SurveyData`."""

from collections.abc import Sequence

import polars as pl

from spectral_radius.surveys.encodings import Encoding, positions_to_pm_1
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


def scale_position(x: pl.Expr, codes: Sequence[int]) -> pl.Expr:
    """Each answer's position on its scale, `0` to `len(codes) - 1`, given its codes."""
    return x.replace_strict(codes, range(len(codes)), return_dtype=pl.Int64)


def wide(
    data: SurveyData,
    survey: str,
    variables: Sequence[str],
    *,
    encoding: Encoding = positions_to_pm_1,
    min_year: int | None = None,
) -> pl.DataFrame:
    """
    One row per respondent with one column per variable, plus demographics and weight `w`.

    Rows are respondents from `min_year` onward (all years if `None`).
    Every non-answer is null, whatever its `missing` kind.
    Each answer is replaced by its position on the variable's scale
    (`0` for the first level, in scale order, whatever its raw code),
    and `encoding` (see `encodings`) is applied to those positions over the rows;
    the default spaces the levels evenly over [-1, 1].
    Every variable must be one the survey reads.
    """
    variables = list(dict.fromkeys(variables))
    codes = declared_scale_codes(data, survey, variables)

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
            *(
                pl.col(v).pipe(scale_position, codes[v]).pipe(encoding, len(codes[v]))
                for v in variables
            ),
            "w",
        )
    )
