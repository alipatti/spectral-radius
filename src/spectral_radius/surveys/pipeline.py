"""Load cleaned data from every source, and write it out for use elsewhere."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl
from loguru import logger

from spectral_radius.surveys.gss import GSS
from spectral_radius.surveys.question_sets import ALLOWED_CATEGORICAL, QUESTION_SETS
from spectral_radius.surveys.schemas import SurveyData
from spectral_radius.surveys.source import DATA_ROOT, Source

SOURCES: tuple[Source, ...] = (GSS(),)
MEMBERS = ("codebook", "respondents", "responses")


def surveys() -> list[str]:
    """
    The names of the surveys of `SOURCES`, in order, for the analysis to iterate over.

    Raises unless every survey has a question set and an entry in
    `ALLOWED_CATEGORICAL`, and each of those belongs to a survey.
    """
    names = [s.name for s in SOURCES]

    if len(set(names)) != len(names):
        raise ValueError(f"Sources share names: {names}")

    for config, keys in {
        "QUESTION_SETS": QUESTION_SETS,
        "ALLOWED_CATEGORICAL": ALLOWED_CATEGORICAL,
    }.items():
        if missing := [n for n in names if n not in keys]:
            raise ValueError(f"Surveys {missing} have no entry in {config}")

        if extra := [k for k in keys if k not in names]:
            raise ValueError(
                f"{config} has entries for surveys without sources: {extra}"
            )

    return names


def source(survey: str) -> Source:
    """The source of `survey` among `SOURCES`."""
    by_name = {s.name: s for s in SOURCES}

    if survey not in by_name:
        raise ValueError(f"No source named {survey!r} (sources {list(by_name)})")

    return by_name[survey]


def load(
    sources: Sequence[Source] = SOURCES,
    *,
    refresh: bool = False,
) -> SurveyData:
    """
    Clean and validate the raw data of `sources`, downloading it if not cached.

    Cleaning reruns every time (rather than being cached) so the result
    can never be stale with respect to the cleaning code.
    """
    return SurveyData.concat(source.load(refresh=refresh) for source in sources)


def build(
    sources: Sequence[Source] = SOURCES,
    dest: Path = DATA_ROOT / "clean",
    *,
    refresh: bool = False,
) -> SurveyData:
    """Load the data (redownloading it if `refresh`) and write each frame to `dest`."""
    data = load(sources, refresh=refresh)
    dest.mkdir(parents=True, exist_ok=True)

    for member in MEMBERS:
        getattr(data, member).write_parquet(dest / f"{member}.parquet")

    # the demographics each survey declares, so a survey's all-null columns
    # read as "not asked" rather than "unanswered" outside python too
    pl.DataFrame(
        [
            {"survey": survey, "demographic": demographic}
            for survey, demographics in data.demographics.items()
            for demographic in sorted(demographics)
        ],
        schema={"survey": pl.String, "demographic": pl.String},
    ).write_parquet(dest / "demographics.parquet")

    logger.info("Wrote clean data to {}", dest)
    return data
