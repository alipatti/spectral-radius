"""The contract every survey source implements, and the caching they share."""

import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, NamedTuple

import polars as pl
from loguru import logger

from spectral_radius.surveys.schemas import SurveyData

DATA_ROOT = Path("data")

type Frame = pl.DataFrame | pl.LazyFrame


class CleanFrames(NamedTuple):
    """A source's cleaned frames, before validation and without a `survey` column."""

    codebook: Frame
    respondents: Frame
    responses: Frame


class Source(ABC):
    """
    A survey: how to download its raw files and clean them into `SurveyData`.

    Subclasses implement `download` and `clean`;
    `fetch` caches the raw files under `<root>/raw/<name>/`
    and `load` validates the cleaned frames.
    """

    name: ClassVar[str]

    def __init__(self, root: Path = DATA_ROOT) -> None:
        self.root = root

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw" / self.name

    @abstractmethod
    def download(self, dest: Path) -> None:
        """Write the raw files into the empty directory `dest`."""

    @abstractmethod
    def clean(self, raw: Path) -> CleanFrames:
        """Clean the raw files in `raw` into frames matching the schemas."""

    def fetch(self, *, refresh: bool = False) -> Path:
        """Return the directory of raw files, downloading them if not cached."""
        if self.raw_dir.exists() and not refresh:
            return self.raw_dir

        self.raw_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{self.name}-", dir=self.raw_dir.parent)
        )

        try:
            logger.info("Downloading {} into {}", self.name, staging)
            self.download(staging)
            self._replace_raw_dir(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return self.raw_dir

    def _replace_raw_dir(self, staging: Path) -> None:
        # a directory can't be renamed over a non-empty one, so move the old one
        # aside first; each rename is atomic, so the cache is never half-written
        stale = self.raw_dir.with_name(f".{self.name}-stale")
        shutil.rmtree(stale, ignore_errors=True)

        if self.raw_dir.exists():
            self.raw_dir.rename(stale)

        staging.rename(self.raw_dir)
        shutil.rmtree(stale, ignore_errors=True)

    def load(self, *, refresh: bool = False) -> SurveyData:
        """Fetch, clean, and validate this source's data."""
        survey = pl.lit(self.name, dtype=pl.Categorical).alias("survey")
        frames = self.clean(self.fetch(refresh=refresh))

        return SurveyData.validate(
            **{
                member: frame.lazy().select(survey, pl.all())
                for member, frame in frames._asdict().items()
            }
        )
