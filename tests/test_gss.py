from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import polars.selectors as cs
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from spectral_radius.surveys import DEMOGRAPHICS, GSS, Respondents
from spectral_radius.surveys.gss import (
    QUESTIONS_SCHEMA,
    VARIABLE_SCALES,
    VARIABLES_SCHEMA,
    codebook,
    respondents,
    responses,
    variables_table,
)
from spectral_radius.surveys.question_sets import question_variables
from spectral_radius.surveys.scales import COURTS_HARSHNESS, YES_NO, Scale, level
from spectral_radius.surveys.schemas import MISSING

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
# yes and no, with "maybe" (3) and "other" (5) off the scale
YES_NO_MAYBE = Scale(
    "yes_no_maybe",
    (level(1, "yes"), level(2, "no")),
    off_scale=(level(5, "other"), level(3, "maybe")),
)
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


def test_responses_keep_every_respondent_and_variable():
    """Give a row per respondent and variable, with a kind for each non-answer."""
    raw = raw_gss({"sex": [1, 2, 1]}).with_columns(
        q=pl.Series([1, None, None], dtype=pl.Int8),
        q_null=pl.Series([None, ".d", ".i"]),
    )

    assert_frame_equal(
        responses(raw, {"q": YES_NO}).collect(),
        pl.DataFrame(
            {
                "year": [2000] * 3,
                "respondent": [1, 2, 3],
                "variable": ["q"] * 3,
                "value": [1, None, None],
                "missing": [None, "dont_know", "inapplicable"],
            },
            schema_overrides={"variable": pl.Categorical, "missing": MISSING},
        ),
    )


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        (".d", "dont_know"),
        (".n", "no_answer"),
        (".i", "inapplicable"),
        (".j", "inapplicable"),
        (".m", "unspecified"),
        (".y", "not_asked_this_year"),
        (".z", "other"),
        (None, "unrecorded"),
    ],
)
def test_responses_map_missing_codes_to_kinds(code: str | None, kind: str):
    """Map each missing code to its kind, and a bare null to `unrecorded`."""
    raw = raw_gss({"sex": [1]}).with_columns(
        q=pl.Series([None], dtype=pl.Int8),
        q_null=pl.Series([code], dtype=pl.String),
    )

    assert responses(raw, {"q": YES_NO}).collect()["missing"].to_list() == [kind]


def test_responses_reject_unknown_missing_code():
    """Fail on a missing code with no kind, rather than guessing one."""
    raw = raw_gss({"sex": [1]}).with_columns(
        q=pl.Series([None], dtype=pl.Int8),
        q_null=pl.Series([".q"]),
    )

    with pytest.raises(pl.exceptions.InvalidOperationError):
        responses(raw, {"q": YES_NO}).collect()


def test_codebook_years_and_labels():
    """List the years each variable was asked and pair values with labels."""
    raw = pl.LazyFrame({"year": [2000, 2002, 2004], "q": [1, None, 2]})
    variables = pl.LazyFrame(
        {
            "variable": ["year", "q"],
            "description": ["Year", "A question"],
            "values": [[], [1, 2]],
            "labels": [[], ["yes", "no"]],
            "missing_codes": [[], [".d", ".i"]],
            "missing_labels": [[], ["don't know", "iap"]],
        },
        schema=VARIABLES_SCHEMA,
    )

    q = (
        codebook(raw, variables, questions([]), {"q": YES_NO})
        .collect()
        .row(1, named=True)
    )

    assert q["years"] == [2000, 2004]
    assert q["labels"] == {"value": [1, 2], "label": ["yes", "no"]}
    assert q["missing"] == {
        "code": [".d", ".i"],
        "label": ["don't know", "iap"],
        "kind": ["dont_know", "inapplicable"],
    }


def test_codebook_lists_off_scale_codes_after_missing_codes():
    """List each labeled off-scale code of the scale as a missing code, in order."""
    raw = pl.LazyFrame({"year": [2000], "q": [1]})
    variables = pl.LazyFrame(
        {
            "variable": ["q"],
            "description": ["A question"],
            "values": [[1, 2, 3, 5]],
            "labels": [["yes", "no", "maybe", "other"]],
            "missing_codes": [[".d"]],
            "missing_labels": [["don't know"]],
        },
        schema=VARIABLES_SCHEMA,
    )

    q = codebook(raw, variables, questions([]), {"q": YES_NO_MAYBE}).collect()

    assert q.row(0, named=True)["missing"] == {
        "code": [".d", "3", "5"],
        "label": ["don't know", "maybe", "other"],
        "kind": ["dont_know", "off_scale", "off_scale"],
    }


def test_responses_move_off_scale_codes_out_of_value():
    """Read an off-scale code as a non-answer of kind `off_scale`."""
    raw = raw_gss({"sex": [1, 1, 1]}).with_columns(
        q=pl.Series([1, 5, None], dtype=pl.Int8),
        q_null=pl.Series([None, None, ".d"]),
    )

    out = responses(raw, {"q": YES_NO_MAYBE}).collect()

    assert out.select("value", "missing").rows() == [
        (1, None),
        (None, "off_scale"),
        (None, "dont_know"),
    ]


def test_responses_keep_undeclared_codes_as_values():
    """Keep a code neither on the scale nor off it, so validation rejects it."""
    raw = raw_gss({"sex": [1]}).with_columns(
        q=pl.Series([7], dtype=pl.Int8),
        q_null=pl.Series([None], dtype=pl.String),
    )

    out = responses(raw, {"q": YES_NO_MAYBE}).collect()

    assert out.select("value", "missing").rows() == [(7, None)]


def test_codebook_marks_variables_read_with_their_scales():
    """Mark exactly the variables read, giving only them a scale and its codes."""
    raw = pl.LazyFrame({"year": [2000], "q": [1], "r": [2]})
    variables = pl.LazyFrame(
        [{"variable": v} for v in ["year", "q", "r"]],
        schema=VARIABLES_SCHEMA,
    ).with_columns(cs.list().fill_null([]))
    read = {"q": COURTS_HARSHNESS, "r": YES_NO}

    result = codebook(raw, variables, questions([]), read).collect()

    assert result.select("in_responses", "scale", "scale_codes").rows() == [
        (False, None, None),
        (True, "courts_harshness", [1, 3, 2]),
        (True, "yes_no", [1, 2]),
    ]


def test_rejects_variable_without_assigned_scale():
    """Refuse to read a variable that isn't assigned a scale."""
    with pytest.raises(ValueError, match="without an assigned scale"):
        GSS(variables=["abany", "undeclared"])


def test_assigns_scale_to_every_question_set_variable():
    """Assign a scale to every variable the question sets name."""
    assert set(question_variables("gss")) <= VARIABLE_SCALES.keys()


def test_codebook_question_text_matches_differently_named_variables():
    """Match the explorer's upper-case and underscored names to the data's."""
    raw = pl.LazyFrame({"year": [2000], "id": [1], "racdif1": [1], "q": [1]})
    variables = pl.LazyFrame(
        [{"variable": v} for v in ["id", "racdif1", "q"]],
        schema=VARIABLES_SCHEMA,
    ).with_columns(cs.list().fill_null([]))
    published = questions([("id_", "Respondent id"), ("RACDIF1", "Differences?")])

    result = codebook(raw, variables, published, {}).collect()

    assert result["question"].to_list() == ["Respondent id", "Differences?", None]


def test_declares_every_demographic():
    """Declare all the shared demographics, since the GSS asks about each."""
    assert GSS.demographics == DEMOGRAPHICS.keys()


def test_variables_table_splits_missing_codes_from_values():
    """Sort integer value labels apart from the labels of extended missing values."""
    metadata = [
        {
            "name": "q",
            "label": "A question",
            "value_labels": {
                "2": "no",
                "MISSING_i": "iap",
                "1": "yes",
                "MISSING_d": "don't know",
            },
        },
        {"name": "id", "label": None, "value_label_name": None},
    ]

    assert variables_table(metadata).to_dicts() == [
        {
            "variable": "q",
            "description": "A question",
            "values": [1, 2],
            "labels": ["yes", "no"],
            "missing_codes": [".d", ".i"],
            "missing_labels": ["don't know", "iap"],
        },
        {
            "variable": "id",
            "description": None,
            "values": [],
            "labels": [],
            "missing_codes": [],
            "missing_labels": [],
        },
    ]


def test_clean_rejects_raw_data_without_missing_codes(tmp_path: Path):
    """Refuse raw data cached before missing codes were kept, naming the fix."""
    pl.DataFrame({"variable": ["q"], "values": [[1]]}).write_parquet(
        tmp_path / "variables.parquet"
    )

    with pytest.raises(ValueError, match="refresh=True"):
        GSS(root=tmp_path).clean(tmp_path)
