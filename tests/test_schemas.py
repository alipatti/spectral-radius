from typing import Any

import polars as pl
import pytest
from dataframely.exc import SchemaError, ValidationError
from helpers import codebook_rows, frame, survey_data

from spectral_radius.surveys import (
    DEMOGRAPHICS,
    Codebook,
    Respondents,
    Responses,
    SurveyData,
)

RESPONDENT = {"survey": "fake", "year": 2000, "respondent": 1, "weight": 1.0}
RESPONSE = {
    "survey": "fake",
    "year": 2000,
    "respondent": 1,
    "variable": "q",
    "value": 1,
}


def codebook_row(**overrides: Any) -> dict[str, Any]:
    return codebook_rows("fake", ["q"])[0] | overrides


def test_demographics_are_respondent_columns_besides_key_and_weight():
    """Map each non-key, non-weight respondent column to its dtype, in schema order."""
    schema = Respondents.create_empty().schema
    expected = {
        name: dtype
        for name, dtype in schema.items()
        if name not in {"survey", "year", "respondent", "weight"}
    }

    assert list(DEMOGRAPHICS.items()) == list(expected.items())


def test_valid_data_passes():
    """Accept consistent frames."""
    data = survey_data([RESPONDENT], [RESPONSE])

    assert len(data.responses) == 1


def test_rejects_duplicate_respondent():
    """Reject two rows for the same respondent in a survey-year."""
    with pytest.raises(ValidationError, match="primary_key"):
        Respondents.validate(frame(Respondents, [RESPONDENT, RESPONDENT]))


def test_rejects_nonpositive_weight():
    """Reject respondents with zero weight."""
    with pytest.raises(ValidationError, match="weight"):
        Respondents.validate(frame(Respondents, [RESPONDENT | {"weight": 0.0}]))


@pytest.mark.parametrize(
    ("row", "rule"),
    [
        (
            codebook_row(labels={"value": [1, 2], "label": ["yes"]}),
            "labels_aligned",
        ),
        (
            codebook_row(labels={"value": [2, 1], "label": ["no", "yes"]}),
            "label_values_sorted_unique",
        ),
        (codebook_row(years=[2000, 2000]), "years_sorted_unique"),
        (codebook_row(years=[2002, 2000]), "years_sorted_unique"),
    ],
)
def test_rejects_bad_codebook(row: dict[str, Any], rule: str):
    """Reject misaligned labels and unsorted or repeated years."""
    with pytest.raises(ValidationError, match=rule):
        Codebook.validate(frame(Codebook, [row]))


def test_rejects_category_outside_enum():
    """Reject a demographic value that isn't one of the shared categories."""
    schema = Respondents.create_empty().schema | {"sex": pl.String}
    martian = pl.DataFrame([RESPONDENT | {"sex": "Martian"}], schema=schema)

    with pytest.raises(ValidationError, match="sex"):
        Respondents.validate(martian, cast=True)


def test_rejects_uncast_categories():
    """Require sources to produce enums, not strings."""
    schema = Respondents.create_empty().schema | {"sex": pl.String}

    with pytest.raises(SchemaError, match="sex"):
        SurveyData.validate(
            codebook=frame(Codebook, [codebook_row()]),
            respondents=pl.DataFrame([RESPONDENT | {"sex": "Male"}], schema=schema),
            responses=frame(Responses, [RESPONSE]),
        )


def test_rejects_response_from_unknown_respondent():
    """Reject responses whose respondent isn't in Respondents."""
    with pytest.raises(ValueError, match="respondents missing"):
        SurveyData.validate(
            codebook=frame(Codebook, [codebook_row()]),
            respondents=frame(Respondents, [RESPONDENT]),
            responses=frame(Responses, [RESPONSE | {"respondent": 2}]),
        )


def test_rejects_response_to_unknown_variable():
    """Reject responses to variables missing from the codebook."""
    with pytest.raises(ValueError, match="variables missing"):
        SurveyData.validate(
            codebook=frame(Codebook, [codebook_row()]),
            respondents=frame(Respondents, [RESPONDENT]),
            responses=frame(Responses, [RESPONSE | {"variable": "other"}]),
        )


def test_concat_stacks_surveys():
    """Combine data from different surveys."""
    other = {"survey": "other"}
    data = SurveyData.concat(
        [
            survey_data([RESPONDENT], [RESPONSE]),
            survey_data([RESPONDENT | other], [RESPONSE | other]),
        ]
    )

    assert data.respondents["survey"].cast(pl.String).to_list() == ["fake", "other"]


def test_concat_rejects_overlapping_surveys():
    """Reject concatenating the same respondents twice."""
    data = survey_data([RESPONDENT], [RESPONSE])

    with pytest.raises(ValidationError, match="primary_key"):
        SurveyData.concat([data, data])
