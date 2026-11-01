from typing import Any

import polars as pl
import pytest
from dataframely.exc import SchemaError, ValidationError
from helpers import codebook_rows, declare_all, frame, survey_data

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
        (
            codebook_row(
                missing={"code": [".d", ".i"], "label": ["dk"], "kind": ["dont_know"]}
            ),
            "missing_aligned",
        ),
        (
            codebook_row(
                missing={
                    "code": [".d", ".d"],
                    "label": ["dk", "can't choose"],
                    "kind": ["dont_know", "dont_know"],
                }
            ),
            "missing_codes_unique",
        ),
        (codebook_row(years=[2000, 2000]), "years_sorted_unique"),
        (codebook_row(years=[2002, 2000]), "years_sorted_unique"),
    ],
)
def test_rejects_bad_codebook(row: dict[str, Any], rule: str):
    """Reject misaligned labels or missing codes, and unsorted or repeated years."""
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
            demographics=declare_all([RESPONDENT]),
        )


def test_rejects_response_from_unknown_respondent():
    """Reject responses whose respondent isn't in Respondents."""
    with pytest.raises(ValueError, match="respondents missing"):
        SurveyData.validate(
            codebook=frame(Codebook, [codebook_row()]),
            respondents=frame(Respondents, [RESPONDENT]),
            responses=frame(Responses, [RESPONSE | {"respondent": 2}]),
            demographics=declare_all([RESPONDENT]),
        )


def test_rejects_response_to_unknown_variable():
    """Reject responses to variables missing from the codebook."""
    with pytest.raises(ValueError, match="variables missing"):
        SurveyData.validate(
            codebook=frame(Codebook, [codebook_row()]),
            respondents=frame(Respondents, [RESPONDENT]),
            responses=frame(Responses, [RESPONSE | {"variable": "other"}]),
            demographics=declare_all([RESPONDENT]),
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


def test_rejects_survey_without_declared_demographics():
    """Require every survey of the respondents to declare its demographics."""
    with pytest.raises(ValueError, match="don't declare"):
        survey_data([RESPONDENT], [RESPONSE], demographics={})


def test_rejects_declared_non_demographic():
    """Reject a declaration naming a column that isn't a demographic."""
    with pytest.raises(ValueError, match="non-demographics"):
        survey_data(
            [RESPONDENT], [RESPONSE], demographics={"fake": frozenset({"weight"})}
        )


def test_concat_keeps_each_surveys_demographics():
    """Keep each survey's declared demographics when stacking."""
    other = {"survey": "other"}
    data = SurveyData.concat(
        [
            survey_data([RESPONDENT], [RESPONSE], {"fake": frozenset({"sex"})}),
            survey_data(
                [RESPONDENT | other],
                [RESPONSE | other],
                {"other": frozenset({"race"})},
            ),
        ]
    )

    assert data.demographics == {"fake": {"sex"}, "other": {"race"}}


def test_require_demographic_rejects_undeclared():
    """Refuse a demographic the survey doesn't provide, naming both."""
    data = survey_data([RESPONDENT], [RESPONSE], {"fake": frozenset({"sex"})})
    data.require_demographic("fake", "sex")

    with pytest.raises(ValueError, match="'fake'.*'race'"):
        data.require_demographic("fake", "race")


@pytest.mark.parametrize(
    "response",
    [
        RESPONSE | {"value": None},
        RESPONSE | {"missing": "dont_know"},
    ],
    ids=["neither", "both"],
)
def test_rejects_response_without_exactly_one_of_value_and_missing(
    response: dict[str, Any],
):
    """Require each response to hold either a value or a missing kind."""
    with pytest.raises(ValidationError, match="value_or_missing"):
        Responses.validate(frame(Responses, [response]))


def test_accepts_listed_missing_kind():
    """Accept a non-answer whose kind the variable's codebook entry lists."""
    data = survey_data([RESPONDENT], [RESPONSE | {"value": None, "missing": "refused"}])

    assert data.responses["missing"].to_list() == ["refused"]


def validate(codebook: dict[str, Any], responses: list[dict[str, Any]]) -> SurveyData:
    return SurveyData.validate(
        codebook=frame(Codebook, [codebook]),
        respondents=frame(Respondents, [RESPONDENT, RESPONDENT | {"respondent": 2}]),
        responses=frame(Responses, responses),
        demographics=declare_all([RESPONDENT]),
    )


def test_rejects_missing_kind_absent_from_codebook():
    """Reject a non-answer whose kind the variable's codebook entry doesn't list."""
    dont_know = RESPONSE | {"value": None, "missing": "dont_know"}

    with pytest.raises(ValueError, match="missing kinds"):
        validate(
            codebook_row(),
            [RESPONSE, dont_know | {"respondent": 2}],
        )


def test_accepts_unrecorded_kind_absent_from_codebook():
    """Allow a missing value without a code, which no codebook can list."""
    unrecorded = RESPONSE | {"respondent": 2, "value": None, "missing": "unrecorded"}

    data = validate(codebook_row(), [RESPONSE, unrecorded])

    assert len(data.responses) == 2


def test_rejects_incomplete_responses():
    """Require a row for every respondent and variable of the survey."""
    with pytest.raises(ValueError, match="lack a row"):
        validate(codebook_row(), [RESPONSE])
