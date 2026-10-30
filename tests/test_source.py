from collections.abc import Sequence
from pathlib import Path
from typing import override

import polars as pl
import pytest
from helpers import codebook_rows, frame
from polars.testing import assert_frame_equal

from spectral_radius.surveys import (
    DEMOGRAPHICS,
    CleanFrames,
    Codebook,
    Responses,
    Source,
    build,
    load,
)
from spectral_radius.surveys.schemas import RACE, SEX

MALE = pl.Series("sex", ["Male"], dtype=SEX)


class FakeSource(Source):
    """
    A source whose 'download' writes a tiny survey, counting each call.

    It declares only `sex`; its one respondent has `demographic_columns`.
    """

    name = "fake"
    demographics = frozenset({"sex"})

    def __init__(
        self,
        root: Path,
        *,
        fail: bool = False,
        demographic_columns: Sequence[pl.Series] = (MALE,),
    ) -> None:
        super().__init__(root)
        self.fail = fail
        self.demographic_columns = demographic_columns
        self.downloads = 0

    @override
    def download(self, dest: Path) -> None:
        self.downloads += 1
        (dest / "download.txt").write_text(str(self.downloads))

        if self.fail:
            raise RuntimeError("Connection dropped")

    @override
    def clean(self, raw: Path) -> CleanFrames:
        def without_survey(schema, rows):
            return frame(schema, [{"survey": "", **row} for row in rows]).drop("survey")

        return CleanFrames(
            codebook=without_survey(Codebook, codebook_rows("", ["q"])),
            respondents=pl.DataFrame(
                {"year": [2000], "respondent": [1], "weight": [1.0]}
            ).with_columns(self.demographic_columns),
            responses=without_survey(
                Responses,
                [{"year": 2000, "respondent": 1, "variable": "q", "value": 3}],
            ),
        )


def test_fetch_downloads_once(tmp_path: Path):
    """Download on the first fetch and reuse the cache afterward."""
    source = FakeSource(tmp_path)

    raw = source.fetch()
    source.fetch()

    assert raw == tmp_path / "raw" / "fake"
    assert (raw / "download.txt").read_text() == "1"
    assert source.downloads == 1


def test_refresh_redownloads(tmp_path: Path):
    """Replace the cached files when refreshing."""
    source = FakeSource(tmp_path)

    source.fetch()
    raw = source.fetch(refresh=True)

    assert (raw / "download.txt").read_text() == "2"
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["fake"]


def test_failed_download_leaves_no_cache(tmp_path: Path):
    """Leave nothing behind when a download raises."""
    with pytest.raises(RuntimeError):
        FakeSource(tmp_path, fail=True).fetch()

    assert list((tmp_path / "raw").iterdir()) == []


def test_failed_refresh_keeps_old_cache(tmp_path: Path):
    """Keep the previous download when a refresh raises."""
    FakeSource(tmp_path).fetch()

    with pytest.raises(RuntimeError):
        FakeSource(tmp_path, fail=True).fetch(refresh=True)

    assert (tmp_path / "raw" / "fake" / "download.txt").read_text() == "1"
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["fake"]


def test_load_adds_survey_column(tmp_path: Path):
    """Label every cleaned frame with the source's name."""
    data = FakeSource(tmp_path).load()

    for member in (data.codebook, data.respondents, data.responses):
        assert member["survey"].cast(pl.String).to_list() == ["fake"]


def test_build_writes_each_frame(tmp_path: Path):
    """Write each validated frame to parquet for use outside python."""
    data = build([FakeSource(tmp_path)], dest=tmp_path / "clean")

    for member in ("codebook", "respondents", "responses"):
        written = pl.read_parquet(tmp_path / "clean" / f"{member}.parquet")
        assert_frame_equal(written, getattr(data, member))


def test_build_writes_declared_demographics(tmp_path: Path):
    """Write which demographics each survey declares, one row per pair."""
    build([FakeSource(tmp_path)], dest=tmp_path / "clean")

    written = pl.read_parquet(tmp_path / "clean" / "demographics.parquet")
    assert written.rows() == [("fake", "sex")]


class OtherSource(FakeSource):
    name = "other"
    demographics = frozenset({"race"})


def test_load_pads_undeclared_demographics_with_typed_nulls(tmp_path: Path):
    """Add each undeclared demographic as a null column of its dtype."""
    respondents = FakeSource(tmp_path).load().respondents
    undeclared = {name: dtype for name, dtype in DEMOGRAPHICS.items() if name != "sex"}

    assert respondents["sex"].to_list() == ["Male"]
    assert respondents.select(list(undeclared)).schema == undeclared
    assert respondents.select(list(undeclared)).null_count().row(0) == (1,) * len(
        undeclared
    )


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ([], r"lacks declared demographics: \['sex'\]"),
        (
            [pl.Series("sex", [None], dtype=SEX)],
            r"entirely null: \['sex'\]",
        ),
        (
            [MALE, pl.Series("race", ["White"], dtype=RACE)],
            r"undeclared demographics: \['race'\]",
        ),
    ],
    ids=["missing", "all_null", "undeclared"],
)
def test_load_rejects_respondents_not_matching_declaration(
    tmp_path: Path,
    columns: list[pl.Series],
    message: str,
):
    """Reject cleaned respondents lacking, nulling, or adding to declared demographics."""
    source = FakeSource(tmp_path, demographic_columns=columns)

    with pytest.raises(ValueError, match=rf"Source 'fake' .*{message}"):
        source.load()


@pytest.mark.parametrize("name", ["height", "weight"])
def test_rejects_declaring_non_demographic(name: str):
    """Refuse at class definition a source declaring a non-demographic."""
    with pytest.raises(TypeError, match=name):

        class _(FakeSource):
            demographics = frozenset({name})


def test_requires_declaration():
    """Refuse at class definition a source that doesn't declare its demographics."""
    with pytest.raises(TypeError, match="doesn't declare"):

        class _(Source):
            pass


def test_declaration_survives_pipeline_concat(tmp_path: Path):
    """Record each source's declared demographics in the combined data."""
    other = OtherSource(
        tmp_path,
        demographic_columns=[pl.Series("race", ["White"], dtype=RACE)],
    )
    data = load([FakeSource(tmp_path), other])

    assert data.demographics == {"fake": {"sex"}, "other": {"race"}}
