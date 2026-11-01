"""The contract every survey source implements, and the caching they share."""

import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, NamedTuple, Never

import polars as pl
from loguru import logger

from spectral_radius.surveys.schemas import DEMOGRAPHICS, SurveyData

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

    Subclasses implement `download` and `clean`
    and declare the `demographics` they provide;
    `fetch` caches the raw files under `<root>/raw/<name>/`
    and `load` validates the cleaned frames.
    Cleaned respondents have only the key, weight, and declared demographics;
    `load` adds the other demographics as nulls.
    """

    name: ClassVar[str]
    demographics: ClassVar[frozenset[str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "demographics"):
            raise TypeError(f"{cls.__name__} doesn't declare its demographics")

        if unknown := cls.demographics.difference(DEMOGRAPHICS):
            raise TypeError(f"{cls.__name__} declares non-demographics {unknown}")

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
        """
        Clean the raw files in `raw` into frames matching the schemas.

        Cleaning drops nothing:
        responses hold a row for every respondent and variable read,
        with each missing code moved out of `value`
        into the `missing` kind the source maps it to
        (`unrecorded` for a missing value without a code).
        The codebook lists each variable's codes, labels, and kinds.
        """
        # NOTE: nothing checks that every missing code was moved out of
        # `value`; for surveys whose missing codes are ordinary integers
        # (e.g. anes' 0, 8, 9) one left in would count as an answer and
        # stretch the observed range that `scale_to_pm_1` uses

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
        frames = frames._replace(
            respondents=self._pad_demographics(frames.respondents.lazy().collect())
        )

        return SurveyData.validate(
            **{
                member: frame.lazy().select(survey, pl.all())
                for member, frame in frames._asdict().items()
            },
            demographics={self.name: self.demographics},
        )

    def _pad_demographics(self, respondents: pl.DataFrame) -> pl.DataFrame:
        """Check the declared demographics are provided, and add the others as nulls."""
        provided = set(respondents.columns).intersection(DEMOGRAPHICS)
        declared = sorted(self.demographics)

        def fail(problem: str, columns: Iterable[str]) -> Never:
            raise ValueError(f"Source {self.name!r} {problem}: {sorted(columns)}")

        if missing := self.demographics - provided:
            fail("lacks declared demographics", missing)

        if undeclared := provided - self.demographics:
            fail("provides undeclared demographics", undeclared)

        all_null = respondents.select(pl.col(declared).is_null().all())

        if empty := [c for c in declared if all_null[c].item()]:
            fail("declares demographics that are entirely null", empty)

        return respondents.with_columns(
            pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in DEMOGRAPHICS.items()
            if name not in self.demographics
        )
