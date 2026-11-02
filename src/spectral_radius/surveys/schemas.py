"""Schemas for cleaned survey data, shared by every source."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

import dataframely as dy
import polars as pl

from spectral_radius.surveys.scales import Scale

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

# kinds of non-answer, shared by every source; each source maps its own missing
# codes onto these, and its codebook keeps the codes and their published labels
MISSING_KINDS: Mapping[str, str] = MappingProxyType(
    {
        "dont_know": "The respondent didn't know or couldn't choose.",
        "no_answer": "The respondent was asked but gave no answer.",
        "refused": "The respondent refused to answer.",
        "skipped": "The respondent skipped the question on a self-completed form.",
        "uncodeable": "The respondent answered, but the answer couldn't be coded.",
        "off_scale": (
            "The respondent gave a coded answer that isn't on the question's scale "
            "(e.g. a volunteered 'depends')."
        ),
        "unspecified": (
            "A non-answer the source doesn't say the kind of "
            "(e.g. the GSS's 'dk, na, iap')."
        ),
        "inapplicable": (
            "The question wasn't asked of this respondent "
            "(e.g. skipped by a filter or on another ballot), "
            "or the respondent said it didn't apply to them."
        ),
        "not_asked_this_year": "The question wasn't asked of anyone that year.",
        "not_released": "The answer was collected but isn't in this release.",
        "not_imputable": "A derived or imputed value couldn't be computed.",
        "other": "A source-specific reason, documented in the source's codebook.",
        "unrecorded": "The source records a missing value without any code.",
    }
)
MISSING = pl.Enum(list(MISSING_KINDS))
# the kind that has no code in the source, so needn't appear in the codebook
UNRECORDED = "unrecorded"


def read_columns(
    variable: pl.Expr,
    read: Mapping[str, Scale],
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    The codebook's `in_responses`, `scale`, and `scale_codes` columns.

    `variable` names each codebook row's variable (as a string),
    and `read` maps each variable read into `Responses` to its declared scale;
    the other variables are null in `scale` and `scale_codes`.
    """

    def lookup(values: Iterable[object], dtype: pl.DataType) -> pl.Expr:
        mapping = dict(zip(read, values, strict=True))
        return variable.replace_strict(mapping, default=None, return_dtype=dtype)

    return (
        variable.is_in(list(read)).alias("in_responses"),
        lookup((s.name for s in read.values()), pl.String()).alias("scale"),
        lookup((s.codes for s in read.values()), pl.List(pl.Int64())).alias(
            "scale_codes"
        ),
    )


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
    `labels` lists the variable's value labels as the source publishes them;
    loading checks those of each variable read against its scale (see `scales`).
    Labels needn't cover every scale code
    (the GSS labels only the endpoints of some scales, e.g. `eqwlth`),
    so they don't define which values are missing.
    `missing` lists the variable's missing codes instead:
    each code as the source spells it (e.g. the GSS's `.d`),
    its published label, and the `MISSING` kind the source maps it to.
    Several codes may share a kind, but each code appears once.
    `in_responses` marks the variables the source reads into `Responses`;
    each of those has the name of the `scale` the source declares for it
    (see `scales`) and the `scale_codes` of its answers, in scale order,
    rather than either being inferred from the data.
    Every value in `Responses` is one of its variable's scale codes.
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
    missing = dy.Struct(
        {
            "code": dy.List(dy.String(), nullable=False),
            "label": dy.List(dy.String(), nullable=False),
            "kind": dy.List(dy.Enum(MISSING.categories), nullable=False),
        },
        nullable=False,
    )
    in_responses = dy.Bool(nullable=False)
    scale = dy.String(nullable=True)
    scale_codes = dy.List(dy.Int64(nullable=False), nullable=True)

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

    @dy.rule()
    def missing_aligned(cls) -> pl.Expr:
        missing = pl.col("missing").struct
        n_codes = missing.field("code").list.len()
        return (n_codes == missing.field("label").list.len()) & (
            n_codes == missing.field("kind").list.len()
        )

    @dy.rule()
    def read_variables_scaled(cls) -> pl.Expr:
        read = pl.col("in_responses")
        return (pl.col("scale").is_not_null() == read) & (
            pl.col("scale_codes").is_not_null() == read
        )

    @dy.rule()
    def scale_codes_unique(cls) -> pl.Expr:
        codes = pl.col("scale_codes")
        return codes.is_null() | (codes.list.n_unique() == codes.list.len())

    @dy.rule()
    def scale_has_two_levels(cls) -> pl.Expr:
        codes = pl.col("scale_codes")
        return codes.is_null() | (codes.list.len() >= 2)

    @dy.rule()
    def missing_codes_unique(cls) -> pl.Expr:
        codes = pl.col("missing").struct.field("code")
        return codes.list.n_unique() == codes.list.len()


class Respondents(dy.Schema):
    """One row per respondent of a survey-year, with harmonized demographics."""

    survey = dy.Categorical(primary_key=True)
    year = dy.Int64(primary_key=True)
    respondent = dy.Int64(primary_key=True)
    # NOTE: weights must be strictly positive, which holds for the gss
    # (minimum about 0.074); other surveys (e.g. anes) may have zero weights
    # for respondents outside a sample, which would need this relaxed
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
    """
    One row per respondent and variable, in long format.

    Each row holds either the answer's `value`
    or, if there is none, the `missing` kind saying why (never both).
    Nothing is dropped:
    every respondent of a survey has a row for each variable
    the survey reads into `Responses`,
    including questions they were never asked (kind `inapplicable`).
    """

    survey = dy.Categorical(primary_key=True)
    year = dy.Int64(primary_key=True)
    respondent = dy.Int64(primary_key=True)
    variable = dy.Categorical(primary_key=True)
    value = dy.Int64(nullable=True)
    missing = dy.Enum(MISSING.categories, nullable=True)

    @dy.rule()
    def value_or_missing(cls) -> pl.Expr:
        return pl.col("value").is_null() != pl.col("missing").is_null()


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


def _require_complete(
    responses: pl.DataFrame,
    respondents: pl.DataFrame,
    codebook: pl.DataFrame,
) -> None:
    """
    Require one response from every respondent to each variable read, and no others.

    The variables read are those the codebook marks `in_responses`.
    Given unique keys and responses only from known respondents
    to variables read, a survey's responses are complete exactly when
    they number its respondents times its variables read.
    """
    read = codebook.filter("in_responses")

    _require_keys_in(
        responses,
        read,
        VARIABLE_KEY,
        "Responses to variables the codebook doesn't mark `in_responses`",
    )

    def count(frame: pl.DataFrame, name: str) -> pl.DataFrame:
        return frame.group_by("survey").agg(pl.len().alias(name))

    incomplete = (
        count(respondents, "respondents")
        .join(count(read, "variables"), on="survey", how="full", coalesce=True)
        .join(count(responses, "rows"), on="survey", how="full", coalesce=True)
        .fill_null(0)
        .filter(pl.col("rows") != pl.col("variables") * pl.col("respondents"))
    )

    if len(incomplete):
        raise ValueError(
            "Responses lack a row for some respondent and variable read "
            f"(surveys {incomplete['survey'].cast(pl.String).to_list()})"
        )


def _require_missing_kinds_in_codebook(
    responses: pl.DataFrame,
    codebook: pl.DataFrame,
) -> None:
    """Require each response's missing kind to be one its codebook entry lists."""
    listed = codebook.select(
        *VARIABLE_KEY,
        pl.col("missing").struct.field("kind").alias("missing"),
    ).explode("missing", empty_as_null=False)
    used = responses.filter(pl.col("missing") != UNRECORDED).select(
        *VARIABLE_KEY, "missing"
    )

    _require_keys_in(
        used,
        listed,
        (*VARIABLE_KEY, "missing"),
        "Responses with missing kinds their codebook entry doesn't list",
    )


def _require_values_on_scale(
    responses: pl.DataFrame,
    codebook: pl.DataFrame,
) -> None:
    """Require every response value to be one of its variable's scale codes."""
    off_scale = (
        responses.filter(pl.col("value").is_not_null())
        .select(*VARIABLE_KEY, "value")
        .unique()
        .join(
            codebook.select(*VARIABLE_KEY, "scale", "scale_codes"),
            on=VARIABLE_KEY,
            validate="m:1",
        )
        .filter(pl.col("value").is_in(pl.col("scale_codes")).not_())
        .group_by(*VARIABLE_KEY, "scale", "scale_codes", maintain_order=True)
        .agg(pl.col("value").sort())
        .sort(VARIABLE_KEY)
    )

    if len(off_scale):
        raise ValueError(
            "Responses with values that aren't codes of their variable's scale: "
            + "; ".join(
                f"{survey}.{variable} has {values}, scale {scale!r} has codes {codes}"
                for survey, variable, scale, codes, values in off_scale.iter_rows()
            )
        )


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
        _require_complete(data.responses, data.respondents, data.codebook)
        _require_missing_kinds_in_codebook(data.responses, data.codebook)
        _require_values_on_scale(data.responses, data.codebook)
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
