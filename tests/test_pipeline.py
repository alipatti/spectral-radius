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
    monkeypatch.setattr(pipeline, "ALLOWED_CATEGORICAL", {"a": set(), "b": set()})


def test_surveys_are_the_sources_in_order(two_surveys: None):
    """Enumerate the surveys of the pipeline's sources, in order."""
    assert pipeline.surveys() == ["a", "b"]


def test_surveys_of_the_real_pipeline():
    """Enumerate the GSS alone, for now."""
    assert pipeline.surveys() == ["gss"]


@pytest.mark.parametrize("config", ["QUESTION_SETS", "ALLOWED_CATEGORICAL"])
def test_surveys_rejects_source_without_config(
    two_surveys: None,
    monkeypatch: pytest.MonkeyPatch,
    config: str,
):
    """Refuse a source without a question set or categorical allowance."""
    monkeypatch.setattr(pipeline, config, {"a": {}})

    with pytest.raises(ValueError, match=rf"\['b'\] have no entry in {config}"):
        pipeline.surveys()


@pytest.mark.parametrize("config", ["QUESTION_SETS", "ALLOWED_CATEGORICAL"])
def test_surveys_rejects_config_without_source(
    two_surveys: None,
    monkeypatch: pytest.MonkeyPatch,
    config: str,
):
    """Refuse a question set or categorical allowance for a survey without a source."""
    monkeypatch.setattr(pipeline, config, {"a": {}, "b": {}, "c": {}})

    with pytest.raises(ValueError, match=rf"{config} .* without sources: \['c'\]"):
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
