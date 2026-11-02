import re

import polars as pl
import pytest

from spectral_radius.surveys.scales import (
    COURTS_HARSHNESS,
    Scale,
    _registry,
    label_mismatches,
    level,
    require_labels_match,
    unlabeled,
)

AGREE = Scale(
    "agree",
    (level(1, "agree", "agree strongly"), unlabeled(2), level(3, "disagree")),
    off_scale=(level(9, "depends"),),
)


def test_courts_levels_are_in_answer_order_not_code_order():
    """Put "about right" (code 3) between "too harshly" and "not harshly enough"."""
    assert COURTS_HARSHNESS.codes == [1, 3, 2]


@pytest.mark.parametrize(
    ("levels", "off_scale", "match"),
    [
        ((level(1, "yes"),), (), "fewer than two levels"),
        ((level(1, "yes"), level(1, "no")), (), r"repeats codes \[1\]"),
        ((level(1, "yes"), level(2, "no")), (level(2, "other"),), r"repeats codes"),
        ((level(1, "yes"), level(2, "no")), (unlabeled(3),), "unlabeled off-scale"),
    ],
    ids=["one_level", "repeated_level", "off_scale_repeats_level", "unlabeled_off"],
)
def test_rejects_bad_scale(levels, off_scale, match: str):
    """Require two or more levels, unique codes, and labeled off-scale codes."""
    with pytest.raises(ValueError, match=match):
        Scale("bad", levels, off_scale)


def test_rejects_labeled_level_without_labels():
    """Require `unlabeled` to declare a level without labels."""
    with pytest.raises(ValueError, match="unlabeled"):
        level(1)


def test_rejects_scales_sharing_a_name():
    """Refuse to register two scales under one name."""
    with pytest.raises(ValueError, match=r"share names \['agree'\]"):
        _registry([AGREE, AGREE])


@pytest.mark.parametrize(("n_levels", "kind"), [(2, "binary"), (3, "ordinal")])
def test_kind_follows_number_of_levels(n_levels: int, kind: str):
    """Call a two-level scale binary and a longer one ordinal."""
    scale = Scale("s", tuple(unlabeled(c) for c in range(n_levels)))

    assert scale.kind == kind


@pytest.mark.parametrize(
    ("values", "labels"),
    [
        ([1, 3], ["agree", "disagree"]),
        ([1, 3], ["Agree  Strongly", " DISAGREE"]),
        ([1, 3, 9], ["agree", "disagree", "depends"]),
    ],
    ids=["exact", "case_and_whitespace", "off_scale"],
)
def test_accepts_matching_labels(values: list[int], labels: list[str]):
    """Accept any listed variant, ignoring case and whitespace, and off-scale codes."""
    assert label_mismatches(AGREE, values, labels) == []


@pytest.mark.parametrize(
    ("values", "labels", "problem"),
    [
        ([1, 3], ["agree", "disagree somewhat"], "code 3 is labeled 'disagree some"),
        ([1, 3, 9], ["agree", "disagree", "other"], "code 9 is labeled 'other'"),
        ([1, 3, 4], ["agree", "disagree", "maybe"], "code 4 .* isn't on scale"),
        ([1, 2, 3], ["agree", "neither", "disagree"], "code 2 is labeled 'neither'"),
        ([1], ["agree"], "code 3 is unlabeled"),
    ],
    ids=["level_label", "off_scale_label", "undeclared", "unlabeled_level", "unpub"],
)
def test_reports_mismatched_labels(
    values: list[int],
    labels: list[str],
    problem: str,
):
    """
    Report a label no variant matches, a labeled code the scale doesn't declare,
    a label on a level declared unlabeled, and a labeled level left unpublished.
    """
    (mismatch,) = label_mismatches(AGREE, values, labels)

    assert re.search(problem, mismatch)


def codebook(rows: list[tuple[str, bool, list[int], list[str]]]) -> pl.DataFrame:
    """A codebook of `(variable, in_responses, values, labels)` rows."""
    return pl.DataFrame(
        [
            {
                "variable": variable,
                "in_responses": read,
                "labels": {"value": values, "label": labels},
            }
            for variable, read, values, labels in rows
        ]
    )


def test_require_labels_match_names_variable_code_and_labels():
    """Raise naming the variable, the code, and both labels."""
    rows = [
        ("q", True, [1, 3], ["agree", "disagree"]),
        ("r", True, [1, 3], ["agree", "nope"]),
        ("unread", False, [1], ["anything"]),
    ]

    with pytest.raises(ValueError, match=r"r: code 3 is labeled 'nope'.*'disagree'"):
        require_labels_match(codebook(rows), {"q": AGREE, "r": AGREE})


def test_require_labels_match_accepts_matching_codebook():
    """Pass when every variable read is labeled as its scale says."""
    rows = [("q", True, [1, 3], ["agree", "disagree"]), ("x", False, [], [])]

    require_labels_match(codebook(rows), {"q": AGREE})


def test_require_labels_match_rejects_variables_without_scales():
    """Raise when the variables read and the variables scaled differ."""
    rows = [("q", True, [1, 3], ["agree", "disagree"])]

    with pytest.raises(ValueError, match=r"read only \['q'\], scaled only \['r'\]"):
        require_labels_match(codebook(rows), {"r": AGREE})
