"""Builders for small validated frames."""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from spectral_radius.surveys import (
    DEMOGRAPHICS,
    Codebook,
    Respondents,
    Responses,
    SurveyData,
)

type Rows = Sequence[Mapping[str, Any]]


def frame(schema: type[Codebook | Respondents | Responses], rows: Rows) -> pl.DataFrame:
    """A frame with `schema`'s dtypes; columns missing from `rows` are null."""
    return pl.DataFrame(rows, schema=schema.create_empty().schema)


def codebook_rows(survey: str, variables: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "survey": survey,
            "variable": v,
            "years": [],
            "labels": {"value": [], "label": []},
        }
        for v in variables
    ]


def declare_all(respondents: Rows) -> dict[str, frozenset[str]]:
    """Declare every demographic for each survey of `respondents`."""
    return {r["survey"]: frozenset(DEMOGRAPHICS) for r in respondents}


def survey_data(
    respondents: Rows,
    responses: Rows,
    demographics: Mapping[str, frozenset[str]] | None = None,
) -> SurveyData:
    """
    Validated data whose codebook lists exactly the variables answered.

    Each survey declares every demographic unless `demographics` says otherwise.
    """
    variables = {(r["survey"], r["variable"]) for r in responses}

    return SurveyData.validate(
        codebook=frame(
            Codebook,
            [row for s, v in sorted(variables) for row in codebook_rows(s, [v])],
        ),
        respondents=frame(Respondents, respondents),
        responses=frame(Responses, responses),
        demographics=declare_all(respondents) if demographics is None else demographics,
    )
