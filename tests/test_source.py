from pathlib import Path
from typing import override

import polars as pl
import pytest
from helpers import codebook_rows, frame

from spectral_radius.surveys import (
    CleanFrames,
    Codebook,
    Respondents,
    Responses,
    Source,
)


class FakeSource(Source):
    """A source whose 'download' writes a tiny survey, counting each call."""

    name = "fake"

    def __init__(self, root: Path, *, fail: bool = False) -> None:
        super().__init__(root)
        self.fail = fail
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
            respondents=without_survey(
                Respondents,
                [{"year": 2000, "respondent": 1, "weight": 1.0}],
            ),
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

