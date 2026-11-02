"""Two synthetic surveys loaded together, as the GSS and ANES will be."""

import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, override

import numpy as np
import plotnine as pn
import polars as pl
import pytest
from helpers import codebook_rows, frame, unlabeled_scale

from spectral_radius import __main__ as cli
from spectral_radius import analysis, appendix, spectral
from spectral_radius.surveys import (
    CleanFrames,
    Codebook,
    Responses,
    Scale,
    Source,
    SurveyData,
    load,
    pipeline,
    question_sets,
    wide,
)
from spectral_radius.surveys.schemas import RACE, SEX

N_PER_YEAR = 30
KINDS = ("dont_know",)


class SyntheticSource(Source):
    """
    A survey fielded in 2000 and 2002, with random answers to its `scales`.

    It provides one demographic, `demographic`, drawn from `categories`.
    """

    demographics = frozenset()
    start_year = 2000
    scales: ClassVar[Mapping[str, Scale]]
    demographic: ClassVar[str]
    categories: ClassVar[pl.Enum]
    seed: ClassVar[int]

    @override
    def download(self, dest: Path) -> None:
        pass

    @override
    def clean(self, raw: Path) -> CleanFrames:
        rng = np.random.default_rng(self.seed)
        years = [y for y in (2000, 2002) for _ in range(N_PER_YEAR)]
        respondents = pl.DataFrame(
            {
                "year": years,
                "respondent": range(len(years)),
                "weight": rng.uniform(0.5, 2, len(years)),
                self.demographic: pl.Series(
                    rng.choice(self.categories.categories.to_list(), len(years)),
                    dtype=self.categories,
                ),
            }
        )

        def answer(variable: str, scale: Scale) -> list[dict]:
            values = rng.choice(scale.codes, len(years))
            dont_know = rng.random(len(years)) < 0.1

            return [
                {"year": year, "respondent": i, "variable": variable}
                | ({"missing": "dont_know"} if dk else {"value": int(value)})
                for i, (year, value, dk) in enumerate(zip(years, values, dont_know))
            ]

        def without_survey(schema, rows):
            return frame(schema, [{"survey": "", **row} for row in rows]).drop("survey")

        return CleanFrames(
            codebook=without_survey(
                Codebook,
                [
                    row
                    for variable, scale in self.scales.items()
                    for row in codebook_rows("", [variable], KINDS, scale=scale)
                ],
            ),
            respondents=respondents,
            responses=without_survey(
                Responses,
                [r for v, s in self.scales.items() for r in answer(v, s)],
            ),
        )


class SurveyX(SyntheticSource):
    name = "x"
    demographics = frozenset({"sex"})
    scales = MappingProxyType(
        {
            "q": unlabeled_scale("zero_to_ten", range(11)),
            "r": unlabeled_scale("one_two", [1, 2]),
        }
    )
    demographic = "sex"
    categories = SEX
    seed = 0


class SurveyY(SyntheticSource):
    name = "y"
    demographics = frozenset({"race"})
    scales = MappingProxyType(
        {
            "q": unlabeled_scale("one_to_five", range(1, 6)),
            "s": unlabeled_scale("one_to_three", range(1, 4)),
        }
    )
    demographic = "race"
    categories = RACE
    seed = 1


QUESTION_SETS = {"x": {"Both": ["q", "r"]}, "y": {"Both": ["q", "s"]}}


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Source, ...]:
    return SurveyX(tmp_path), SurveyY(tmp_path)


@pytest.fixture
def data(sources: tuple[Source, ...]) -> SurveyData:
    return load(sources)


@pytest.mark.parametrize(
    ("survey", "variables", "scale"),
    [("x", ["q", "r"], (0, 10)), ("y", ["q", "s"], (1, 5))],
)
def test_wide_keeps_only_the_surveys_respondents_and_scale(
    data: SurveyData,
    survey: str,
    variables: list[str],
    scale: tuple[int, int],
):
    """Give each survey's respondents, encoding `q` by that survey's scale."""
    out = wide(data, survey, variables)
    raw = data.responses.filter(pl.col("survey") == survey, pl.col("variable") == "q")
    low, high = scale

    assert len(out) == 2 * N_PER_YEAR
    assert out["q"].to_list() == pytest.approx(
        [
            None if v is None else (v - low) / (high - low) * 2 - 1
            for v in raw["value"].to_list()
        ]
    )


def test_wide_rejects_variable_only_the_other_survey_reads(data: SurveyData):
    """Refuse a variable the other survey reads but this one doesn't."""
    with pytest.raises(ValueError, match=r"'x' doesn't read variables \['s'\]"):
        wide(data, "x", ["q", "s"])


@pytest.mark.parametrize(
    ("survey", "declared", "padded"),
    [("x", "sex", "race"), ("y", "race", "sex")],
)
def test_undeclared_demographic_is_null_for_that_survey_only(
    data: SurveyData,
    survey: str,
    declared: str,
    padded: str,
):
    """Pad each survey's undeclared demographic with nulls, keeping its declared one."""
    out = wide(data, survey, ["q"])

    assert out[padded].null_count() == len(out)
    assert out[declared].null_count() == 0


@pytest.mark.parametrize(("survey", "undeclared"), [("x", "race"), ("y", "sex")])
def test_grouping_by_undeclared_demographic_raises_for_that_survey(
    data: SurveyData,
    monkeypatch: pytest.MonkeyPatch,
    survey: str,
    undeclared: str,
):
    """Refuse to group a survey by a demographic only the other survey declares."""
    monkeypatch.setattr(spectral, "survey_data", lambda: data)

    with pytest.raises(ValueError, match=rf"'{survey}'.*'{undeclared}'"):
        spectral.polarization_over_time_data(undeclared, survey=survey)


def test_completeness_is_checked_per_survey(data: SurveyData):
    """Name the survey lacking a response, even when the other is complete."""
    dropped = pl.col("survey").eq("y") & pl.col("respondent").eq(0)

    with pytest.raises(ValueError, match=r"lack a row .*\['y'\]"):
        SurveyData.validate(
            codebook=data.codebook,
            respondents=data.respondents,
            responses=data.responses.filter(~(dropped & pl.col("variable").eq("q"))),
            demographics=data.demographics,
        )


@pytest.fixture
def pipeline_of_two(
    sources: tuple[Source, ...],
    data: SurveyData,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Make the pipeline's sources the two synthetic surveys, already loaded."""
    monkeypatch.setattr(pipeline, "SOURCES", sources)
    monkeypatch.setattr(pipeline, "QUESTION_SETS", QUESTION_SETS)
    monkeypatch.setattr(pipeline, "ALLOWED_CATEGORICAL", {"x": set(), "y": set()})
    monkeypatch.setattr(spectral, "QUESTION_SETS", QUESTION_SETS)
    monkeypatch.setattr(appendix, "QUESTION_SETS", QUESTION_SETS)
    monkeypatch.setattr(
        analysis,
        "ALLOWED_CATEGORICAL",
        {"x": frozenset(), "y": frozenset()},
    )
    monkeypatch.setattr(question_sets, "QUESTION_SETS", QUESTION_SETS)

    for module in analysis, spectral, appendix:
        monkeypatch.setattr(module, "survey_data", lambda: data)

    cached = (
        analysis.analysis_frame,
        spectral._pooled_measures,
        spectral._within_measures,
    )

    for function in cached:
        function.cache_clear()

    yield

    for function in cached:
        function.cache_clear()


def test_figures_are_drawn_for_each_survey(
    pipeline_of_two: None,
    monkeypatch: pytest.MonkeyPatch,
):
    """Draw each survey's figures under its own directory, by its own demographics."""
    saved: list[Path] = []

    def record(figure: pn.ggplot, path: Path, **_: object) -> None:
        assert isinstance(figure, pn.ggplot)
        saved.append(path)

    monkeypatch.setattr(spectral, "savefig", record)

    spectral.main()

    assert saved == [
        spectral.FIGURES / survey / figure
        for survey, group in [("x", "sex"), ("y", "race")]
        for figure in (
            "pooled/pooled",
            "decompositions/trace_concentration",
            f"decompositions/{group}",
            f"by_group/{group}",
        )
    ]


def test_appendix_is_written_for_each_survey(
    pipeline_of_two: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Write each survey's appendix to its own file, listing its own variables."""
    monkeypatch.setattr(appendix, "OUTPUT", tmp_path / "generated")

    appendix.main()

    written = {p.name: p.read_text() for p in (tmp_path / "generated").iterdir()}
    assert sorted(written) == ["x_appendix.tex", "y_appendix.tex"]
    assert r"\textsc{r}" in written["x_appendix.tex"]
    assert r"\textsc{s}" not in written["x_appendix.tex"]
    assert r"\textsc{s}" in written["y_appendix.tex"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["build", "figures", "appendix"]),
        (["data"], ["build(refresh=False)"]),
        (["data", "--refresh"], ["build(refresh=True)"]),
        (["figures"], ["figures"]),
        (["appendix"], ["appendix"]),
    ],
)
def test_cli_runs_each_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: list[str],
):
    """Run the build, figures, and appendix steps each subcommand names."""
    calls: list[str] = []

    def build(**kwargs: bool) -> None:
        calls.append(
            f"build(refresh={kwargs['refresh']})" if kwargs else "build",
        )

    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "figures", lambda: calls.append("figures"))
    monkeypatch.setattr(appendix, "main", lambda: calls.append("appendix"))
    monkeypatch.setattr(sys, "argv", ["python -m spectral_radius", *argv])

    cli.main()

    assert calls == expected


def test_cli_module_runs():
    """Run `python -m spectral_radius --help` as a module."""
    result = subprocess.run(
        [sys.executable, "-m", "spectral_radius", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "figures" in result.stdout
