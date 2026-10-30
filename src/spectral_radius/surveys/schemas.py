"""Schemas for cleaned survey data, shared by every source."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

import dataframely as dy
import polars as pl

# harmonized demographic categories; every source maps its codes onto these
SEX = pl.Enum(["Male", "Female"])
RACE = pl.Enum(["White", "Black", "Other"])
RELIGION = pl.Enum(["Protestant", "Catholic", "Jewish", "None", "Other"])
AGE_BUCKET = pl.Enum(["18-29", "30-49", "50-64", "65+"])
POLITICAL_PARTY = pl.Enum(["Democrat", "Independent", "Republican"])
POLITICAL_IDEOLOGY = pl.Enum(["Liberal", "Moderate", "Conservative"])
WORKFORCE_STATUS = pl.Enum(["Employed", "Unemployed", "Out of labor force"])
EDUCATION = pl.Enum(["No Bachelor's degree", "Bachelor's degree or more"])
GEOGRAPHIC_REGION = pl.Enum(["Northeast", "Midwest", "South", "West"])
TYPE_OF_NEIGHBORHOOD = pl.Enum(["Urban", "Suburban", "Rural"])

RESPONDENT_KEY = ("survey", "year", "respondent")
VARIABLE_KEY = ("survey", "variable")


def _is_strictly_increasing(lists: pl.Expr) -> pl.Expr:
    return lists.list.eval(pl.element().diff().drop_nulls() > 0).list.all()


class Codebook(dy.Schema):
    """
    One row per variable of a survey.

    `description` is the variable's short label
    and `question` the text of the question asked
    (null where the source publishes none).
    `labels` lists the variable's value labels as the source publishes them,
    as documentation only.
    Labels needn't cover every valid value
    (the GSS labels only the endpoints of some scales, e.g. `eqwlth`),
    so they don't define which values are missing:
    each source nulls out its own missing codes while cleaning,
    and `Responses` holds only the answers that remain.
    """

    survey = dy.Categorical(primary_key=True)
    variable = dy.Categorical(primary_key=True)
    description = dy.String(nullable=True)
    question = dy.String(nullable=True)
    years = dy.List(dy.Int64(), nullable=False)
    labels = dy.Struct(
        {
            "value": dy.List(dy.Int64(), nullable=False),
            "label": dy.List(dy.String(), nullable=False),
        },
        nullable=False,
    )

    @dy.rule()
    def years_sorted_unique(cls) -> pl.Expr:
        return _is_strictly_increasing(pl.col("years"))

    @dy.rule()
    def labels_aligned(cls) -> pl.Expr:
        labels = pl.col("labels").struct
        return labels.field("value").list.len() == labels.field("label").list.len()

    @dy.rule()
    def label_values_sorted_unique(cls) -> pl.Expr:
        return _is_strictly_increasing(pl.col("labels").struct.field("value"))


class Respondents(dy.Schema):
    """One row per respondent of a survey-year, with harmonized demographics."""

    survey = dy.Categorical(primary_key=True)
    year = dy.Int64(primary_key=True)
    respondent = dy.Int64(primary_key=True)
    weight = dy.Float64(nullable=False, min_exclusive=0)
    sex = dy.Enum(SEX.categories, nullable=True)
    race = dy.Enum(RACE.categories, nullable=True)
    religion = dy.Enum(RELIGION.categories, nullable=True)
    age_bucket = dy.Enum(AGE_BUCKET.categories, nullable=True)
    political_party = dy.Enum(POLITICAL_PARTY.categories, nullable=True)
    political_ideology = dy.Enum(POLITICAL_IDEOLOGY.categories, nullable=True)
    workforce_status = dy.Enum(WORKFORCE_STATUS.categories, nullable=True)
    education = dy.Enum(EDUCATION.categories, nullable=True)
    self_or_parent_immigrant = dy.Bool(nullable=True)
    geographic_region = dy.Enum(GEOGRAPHIC_REGION.categories, nullable=True)
    type_of_neighborhood = dy.Enum(TYPE_OF_NEIGHBORHOOD.categories, nullable=True)


# the demographic columns of `Respondents` and their dtypes, in schema order
DEMOGRAPHICS: Mapping[str, pl.DataType] = MappingProxyType(
    {
        name: column.dtype
        for name, column in Respondents.columns().items()
        if name not in {*Respondents.primary_key(), "weight"}
    }
)


class Responses(dy.Schema):
    """One row per non-missing answer, in long format."""

    survey = dy.Categorical(primary_key=True)
    year = dy.Int64(primary_key=True)
    respondent = dy.Int64(primary_key=True)
    variable = dy.Categorical(primary_key=True)
    value = dy.Int64(nullable=False)


def _require_keys_in(
    frame: pl.DataFrame,
    reference: pl.DataFrame,
    keys: Iterable[str],
    message: str,
) -> None:
    keys = list(keys)
    orphans = (
        frame.select(keys).unique().join(reference.select(keys), on=keys, how="anti")
    )

    if len(orphans):
        raise ValueError(f"{message} ({len(orphans)} keys, e.g. {orphans.row(0)})")


def _require_declared(
    respondents: pl.DataFrame,
    demographics: Mapping[str, frozenset[str]],
) -> None:
    for survey, declared in demographics.items():
        if unknown := declared.difference(DEMOGRAPHICS):
            raise ValueError(f"Survey {survey!r} declares non-demographics {unknown}")

    surveys = set(respondents["survey"].cast(pl.String).unique())

    if undeclared := surveys.difference(demographics):
        raise ValueError(f"Surveys {undeclared} don't declare their demographics")


@dataclass(frozen=True)
class SurveyData:
    """
    Validated codebook, respondents, and responses of one or more surveys.

    `demographics` maps each survey to the demographics it provides;
    the others are null throughout its respondents.
    """

    codebook: dy.DataFrame[Codebook]
    respondents: dy.DataFrame[Respondents]
    responses: dy.DataFrame[Responses]
    demographics: Mapping[str, frozenset[str]]

    @classmethod
    def validate(
        cls,
        *,
        codebook: pl.DataFrame | pl.LazyFrame,
        respondents: pl.DataFrame | pl.LazyFrame,
        responses: pl.DataFrame | pl.LazyFrame,
        demographics: Mapping[str, frozenset[str]],
    ) -> Self:
        """Validate each frame against its schema and the keys across frames."""
        data = cls(
            codebook=Codebook.validate(codebook.lazy().collect()),
            respondents=Respondents.validate(respondents.lazy().collect()),
            responses=Responses.validate(responses.lazy().collect()),
            demographics=MappingProxyType(dict(demographics)),
        )

        _require_keys_in(
            data.responses,
            data.respondents,
            RESPONDENT_KEY,
            "Responses from respondents missing from Respondents",
        )
        _require_keys_in(
            data.responses,
            data.codebook,
            VARIABLE_KEY,
            "Responses to variables missing from Codebook",
        )
        _require_declared(data.respondents, data.demographics)

        return data

    @classmethod
    def concat(cls, datas: Iterable[Self]) -> Self:
        """Stack data from several sources and revalidate."""
        datas = list(datas)
        return cls.validate(
            codebook=pl.concat(d.codebook for d in datas),
            respondents=pl.concat(d.respondents for d in datas),
            responses=pl.concat(d.responses for d in datas),
            demographics={
                survey: declared
                for d in datas
                for survey, declared in d.demographics.items()
            },
        )

    def require_demographic(self, survey: str, demographic: str) -> None:
        """Raise unless `survey` provides `demographic`, e.g. before grouping by it."""
        if demographic not in self.demographics[survey]:
            raise ValueError(
                f"Survey {survey!r} doesn't provide demographic {demographic!r}"
            )
