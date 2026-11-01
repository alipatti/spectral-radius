"""Builders for small validated frames."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import polars as pl

from spectral_radius.surveys import (
    DEMOGRAPHICS,
    Codebook,
    Respondents,
    Responses,
    SurveyData,
)
from spectral_radius.surveys.schemas import UNRECORDED, VariableScale

type Rows = Sequence[Mapping[str, Any]]

# the kind given to the responses `survey_data` fills in
FILLER = "no_answer"
# a scale wide enough for any test value, for variables not given their own
WIDE_SCALE = VariableScale(0, 100, "ordinal")


def frame(schema: type[Codebook | Respondents | Responses], rows: Rows) -> pl.DataFrame:
    """A frame with `schema`'s dtypes; columns missing from `rows` are null."""
    return pl.DataFrame(rows, schema=schema.create_empty().schema)


def codebook_rows(
    survey: str,
    variables: Sequence[str],
    kinds: Iterable[str] = (),
    *,
    scale: VariableScale | None = WIDE_SCALE,
) -> list[dict[str, Any]]:
    """
    Codebook rows listing each of `kinds` as a missing code named after itself.

    The variables are read into responses with `scale`, or not read if it's `None`.
    """
    kinds = sorted(kinds)

    return [
        {
            "survey": survey,
            "variable": v,
            "years": [],
            "labels": {"value": [], "label": []},
            "missing": {"code": kinds, "label": kinds, "kind": kinds},
            "in_responses": scale is not None,
            "valid": scale and {"min": scale.min, "max": scale.max},
            "scale": scale and scale.scale,
        }
        for v in variables
    ]


def declare_all(respondents: Rows) -> dict[str, frozenset[str]]:
    """Declare every demographic for each survey of `respondents`."""
    return {r["survey"]: frozenset(DEMOGRAPHICS) for r in respondents}


def complete(respondents: Rows, responses: Rows) -> list[Mapping[str, Any]]:
    """
    `responses` plus a `FILLER` row for each respondent and variable they lack.

    Variables are those `responses` has for each survey.
    """
    given = {
        (r["survey"], r["year"], r["respondent"], r["variable"]) for r in responses
    }
    variables = sorted({(r["survey"], r["variable"]) for r in responses})
    filled = [
        {
            "survey": s,
            "year": p["year"],
            "respondent": p["respondent"],
            "variable": v,
            "missing": FILLER,
        }
        for p in respondents
        for s, v in variables
        if s == p["survey"] and (s, p["year"], p["respondent"], v) not in given
    ]

    return [*responses, *filled]


def survey_data(
    respondents: Rows,
    responses: Rows,
    demographics: Mapping[str, frozenset[str]] | None = None,
    scales: Mapping[tuple[str, str], VariableScale] | None = None,
) -> SurveyData:
    """
    Validated data, with `responses` completed with `FILLER` rows.

    The codebook lists exactly the variables answered,
    each with the missing kinds its responses use
    and its scale in `scales` (keyed by survey and variable), else `WIDE_SCALE`.
    Each survey declares every demographic unless `demographics` says otherwise.
    """
    scales = scales or {}
    responses = complete(respondents, responses)
    kinds: dict[tuple[str, str], set[str]] = {}

    for r in responses:
        kind = r.get("missing")
        used = kinds.setdefault((r["survey"], r["variable"]), set())

        if kind is not None and kind != UNRECORDED:
            used.add(kind)

    return SurveyData.validate(
        codebook=frame(
            Codebook,
            [
                row
                for (s, v), used in sorted(kinds.items())
                for row in codebook_rows(
                    s, [v], used, scale=scales.get((s, v), WIDE_SCALE)
                )
            ],
        ),
        respondents=frame(Respondents, respondents),
        responses=frame(Responses, responses),
        demographics=declare_all(respondents) if demographics is None else demographics,
    )
