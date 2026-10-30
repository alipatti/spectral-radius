from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from spectral_radius.surveys import Respondents
from spectral_radius.surveys.gss import (
    QUESTIONS_SCHEMA,
    codebook,
    respondents,
    responses,
)

RAW_DEMOGRAPHICS = [
    "sex",
    "race",
    "relig",
    "age",
    "partyid",
    "polviews",
    "wrkstat",
    "degree",
    "educ",
    "born",
    "parborn",
    "region",
    "xnorcsiz",
]
NO_BA = "No Bachelor's degree"
BA = "Bachelor's degree or more"


def questions(texts: Sequence[tuple[str, str]]) -> pl.LazyFrame:
    """The explorer's question text for each `(name, text)`."""
    return pl.LazyFrame(
        [{"name": name, "surveyQuestion": [text]} for name, text in texts],
        schema=QUESTIONS_SCHEMA,
    )


def raw_gss(columns: Mapping[str, Sequence[Any]]) -> pl.LazyFrame:
    """Raw GSS rows with the given columns and every other demographic null."""
    n = len(next(iter(columns.values())))

    return pl.LazyFrame(
        {
            "year": [2000] * n,
            "id": range(1, n + 1),
            "wtssps": [1.0] * n,
            **{c: [None] * n for c in RAW_DEMOGRAPHICS},
            **columns,
        },
        schema_overrides={c: pl.Int64 for c in RAW_DEMOGRAPHICS},
    )


@pytest.mark.parametrize(
    ("demographic", "raw", "expected"),
    [
        ("sex", {"sex": [1, 2, None]}, ["Male", "Female", None]),
        ("race", {"race": [1, 2, 3, None]}, ["White", "Black", "Other", None]),
        (
            "religion",
            # other listed religions are "Other", but a missing one stays missing
            {"relig": [1, 2, 3, 4, 5, 11, 6, 13, None]},
            [
                "Protestant",
                "Catholic",
                "Jewish",
                "None",
                "Other",
                "Protestant",
                "Other",
                "Other",
                None,
            ],
        ),
        (
            "age_bucket",
            {"age": [18, 29, 30, 49, 50, 64, 65, 89, None]},
            ["18-29", "18-29", "30-49", "30-49", "50-64", "50-64", "65+", "65+", None],
        ),
        (
            "political_party",
            {"partyid": [0, 1, 2, 4, 5, 6, 7, None]},
            [
                "Democrat",
                "Democrat",
                "Independent",
                "Independent",
                "Republican",
                "Republican",
                None,
                None,
            ],
        ),
        (
            "political_ideology",
            {"polviews": [1, 3, 4, 5, 7, None]},
            ["Liberal", "Liberal", "Moderate", "Conservative", "Conservative", None],
        ),
        (
            "workforce_status",
            {"wrkstat": [1, 3, 4, 5, 8, None]},
            [
                "Employed",
                "Employed",
                "Unemployed",
                "Out of labor force",
                "Out of labor force",
                None,
            ],
        ),
        (
            "education",
            # by highest degree alone, whatever the years of school (`educ`)
            {
                "degree": [0, 1, 2, 3, 4, 3, 2, None],
                "educ": [16, 12, 16, 16, 20, 2, 18, 16],
            },
            [NO_BA, NO_BA, NO_BA, BA, BA, BA, NO_BA, None],
        ),
        (
            "self_or_parent_immigrant",
            {
                "born": [1, 2, 1, 1, None, None],
                "parborn": [0, 0, 1, 3, 8, 0],
            },
            [False, True, True, False, True, None],
        ),
        (
            "geographic_region",
            {"region": [1, 2, 3, 4, None]},
            ["Northeast", "Midwest", "South", "West", None],
        ),
        (
            "type_of_neighborhood",
            {"xnorcsiz": [1, 2, 3, 6, 7, 10, None]},
            ["Urban", "Urban", "Suburban", "Suburban", "Rural", "Rural", None],
        ),
    ],
)
def test_recodes_demographic(
    demographic: str,
    raw: Mapping[str, Sequence[Any]],
    expected: Sequence[Any],
):
    """Map GSS codes onto the shared categories."""
    recoded = respondents(raw_gss(raw)).collect()[demographic]
    dtype = Respondents.create_empty().schema[demographic]

    assert_series_equal(recoded, pl.Series(demographic, expected, dtype=dtype))


def test_respondents_match_schema():
    """Produce every Respondents column but the survey, with its dtype."""
    schema = Respondents.create_empty().schema

    assert respondents(raw_gss({"sex": [1]})).collect_schema() == {
        k: v for k, v in schema.items() if k != "survey"
    }


def test_responses_are_long_without_nulls():
    """Keep one row per non-null answer."""
    raw = raw_gss({"sex": [1, 2]}).with_columns(
        q1=pl.Series([1, None]),
        q2=pl.Series([None, None], dtype=pl.Int64),
    )

    assert_frame_equal(
        responses(raw, ["q1", "q2"]).collect(),
        pl.DataFrame(
            {"year": [2000], "respondent": [1], "variable": ["q1"], "value": [1]},
            schema_overrides={"variable": pl.Categorical},
        ),
    )


def test_codebook_years_and_labels():
    """List the years each variable was asked and pair values with labels."""
    raw = pl.LazyFrame({"year": [2000, 2002, 2004], "q": [1, None, 2]})
    variables = pl.LazyFrame(
        {
            "variable": ["year", "q"],
            "description": ["Year", "A question"],
            "values": [[], [1, 2]],
            "labels": [[], ["yes", "no"]],
        },
        schema_overrides={"values": pl.List(pl.Int64), "labels": pl.List(pl.String)},
    )

    q = codebook(raw, variables, questions([])).collect().row(1, named=True)

    assert q["years"] == [2000, 2004]
    assert q["labels"] == {"value": [1, 2], "label": ["yes", "no"]}


def test_codebook_question_text_matches_differently_named_variables():
    """Match the explorer's upper-case and underscored names to the data's."""
    raw = pl.LazyFrame({"year": [2000], "id": [1], "racdif1": [1], "q": [1]})
    variables = pl.LazyFrame(
        {
            "variable": ["id", "racdif1", "q"],
            "description": [None] * 3,
            "values": [[]] * 3,
            "labels": [[]] * 3,
        },
        schema_overrides={
            "description": pl.String,
            "values": pl.List(pl.Int64),
            "labels": pl.List(pl.String),
        },
    )
    published = questions([("id_", "Respondent id"), ("RACDIF1", "Differences?")])

    result = codebook(raw, variables, published).collect()

    assert result["question"].to_list() == ["Respondent id", "Differences?", None]
