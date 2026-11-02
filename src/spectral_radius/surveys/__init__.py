"""
Cleaned survey data in a shared format, from every source.

Each survey is a `Source` (see `source.py`) listed in `pipeline.SOURCES`,
with a question set in `QUESTION_SETS`.

NOTE: planned sources.

ANES (the cumulative data file) is next:
- getting the file: it normally requires accepting terms on electionstudies.org,
  so decide whether `download` fetches it or reads a copy downloaded by hand
- missing codes are ordinary integers (typically 0 inapplicable,
  8 don't know, 9 no answer), so map each explicitly to a `MISSING` kind;
  an unmapped code would otherwise pass as an answer
- weights: `Respondents` requires strictly positive weights,
  which anes may not satisfy (respondents outside a sample get 0);
  relax the rule only once real anes weights show it's needed
- declare `demographics` (likely a subset of `DEMOGRAPHICS`) and `start_year`;
  waves are every 2 or 4 years, which `binned_year` handles but worth checking
- assign each variable a scale in `scales.py`, reusing shared scales
  (agree/disagree, favor/oppose, spending) where the labels match

Michigan surveys of consumers (consumer sentiment) may follow:
- it's monthly, so it needs a time unit finer than `year`
  (e.g. a `month` column on `Respondents`), deliberately deferred until then
- its demographics are sparser, so its `demographics` declaration will be small
"""

from spectral_radius.surveys.gss import GSS
from spectral_radius.surveys.pipeline import SOURCES, build, load
from spectral_radius.surveys.question_sets import QUESTION_SETS, question_variables
from spectral_radius.surveys.scales import SCALES, Scale
from spectral_radius.surveys.schemas import (
    DEMOGRAPHICS,
    Codebook,
    Respondents,
    Responses,
    SurveyData,
)
from spectral_radius.surveys.source import CleanFrames, Source
from spectral_radius.surveys.wide import wide

__all__ = [
    "DEMOGRAPHICS",
    "GSS",
    "QUESTION_SETS",
    "SCALES",
    "SOURCES",
    "CleanFrames",
    "Codebook",
    "Respondents",
    "Responses",
    "Scale",
    "Source",
    "SurveyData",
    "build",
    "load",
    "question_variables",
    "wide",
]
