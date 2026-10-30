"""Load cleaned data from every source, and write it out for use elsewhere."""

from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from spectral_radius.surveys.gss import GSS
from spectral_radius.surveys.schemas import SurveyData
from spectral_radius.surveys.source import DATA_ROOT, Source

SOURCES: tuple[Source, ...] = (GSS(),)
MEMBERS = ("codebook", "respondents", "responses")


def load(sources: Sequence[Source] = SOURCES) -> SurveyData:
    """
    Clean and validate the cached raw data of `sources`.

    Cleaning reruns every time (rather than being cached) so the result
    can never be stale with respect to the cleaning code.
    """
    return SurveyData.concat(source.load() for source in sources)


def build(
    sources: Sequence[Source] = SOURCES,
    dest: Path = DATA_ROOT / "clean",
) -> SurveyData:
    """Load the data and write each frame to `dest` as parquet."""
    data = load(sources)
    dest.mkdir(parents=True, exist_ok=True)

    for member in MEMBERS:
        getattr(data, member).write_parquet(dest / f"{member}.parquet")

    logger.info("Wrote clean data to {}", dest)
    return data
