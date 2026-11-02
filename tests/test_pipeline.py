from pathlib import Path
from types import SimpleNamespace

import pytest

from spectral_radius import appendix
from spectral_radius.surveys import pipeline


@pytest.fixture
def two_surveys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the pipeline's sources two surveys, `a` and `b`, each fully configured."""
    monkeypatch.setattr(
        pipeline,
        "SOURCES",
        (SimpleNamespace(name="a"), SimpleNamespace(name="b")),
    )
    monkeypatch.setattr(pipeline, "QUESTION_SETS", {"a": {}, "b": {}})


def test_surveys_are_the_sources_in_order(two_surveys: None):
    """Enumerate the surveys of the pipeline's sources, in order."""
    assert pipeline.surveys() == ["a", "b"]


def test_surveys_of_the_real_pipeline():
    """Enumerate the GSS alone, for now."""
    assert pipeline.surveys() == ["gss"]


def test_surveys_rejects_source_without_question_set(
    two_surveys: None,
    monkeypatch: pytest.MonkeyPatch,
):
    """Refuse a source without a question set."""
    monkeypatch.setattr(pipeline, "QUESTION_SETS", {"a": {}})

    with pytest.raises(ValueError, match=r"\['b'\] have no entry in QUESTION_SETS"):
        pipeline.surveys()


def test_surveys_rejects_question_set_without_source(
    two_surveys: None,
    monkeypatch: pytest.MonkeyPatch,
):
    """Refuse a question set for a survey without a source."""
    monkeypatch.setattr(pipeline, "QUESTION_SETS", {"a": {}, "b": {}, "c": {}})

    with pytest.raises(ValueError, match=r"QUESTION_SETS .* without sources: \['c'\]"):
        pipeline.surveys()


def test_surveys_rejects_sources_sharing_a_name(
    two_surveys: None,
    monkeypatch: pytest.MonkeyPatch,
):
    """Refuse two sources of the same survey."""
    monkeypatch.setattr(
        pipeline,
        "SOURCES",
        (SimpleNamespace(name="a"), SimpleNamespace(name="a")),
    )

    with pytest.raises(ValueError, match="share names"):
        pipeline.surveys()


def test_gss_appendix_path_is_unchanged():
    """Keep the path the paper inputs the GSS appendix from."""
    assert appendix.appendix_path("gss") == Path("paper/generated/gss_appendix.tex")
