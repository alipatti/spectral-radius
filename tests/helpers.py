"""Builders for small validated frames."""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from spectral_radius.surveys import Codebook, Respondents, Responses, SurveyData

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


def survey_data(respondents: Rows, responses: Rows) -> SurveyData:
    """Validated data whose codebook lists exactly the variables answered."""
    variables = {(r["survey"], r["variable"]) for r in responses}

    return SurveyData.validate(
        codebook=frame(
            Codebook,
            [row for s, v in sorted(variables) for row in codebook_rows(s, [v])],
        ),
        respondents=frame(Respondents, respondents),
        responses=frame(Responses, responses),
    )
