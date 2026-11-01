"""
The General Social Survey.

Codes are documented at https://gssdataexplorer.norc.org/variables/vfilter.
"""

import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, override
from zipfile import ZipFile

import httpx
import polars as pl
import polars.selectors as cs
import polars_readstat as prs

from spectral_radius.surveys import schemas
from spectral_radius.surveys.question_sets import question_variables
from spectral_radius.surveys.source import DATA_ROOT, CleanFrames, Source

URL = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip"
# the search api behind the gss data explorer, the only source of question text
QUESTIONS_URL = (
    "https://3ilfsaj2lj.execute-api.us-east-1.amazonaws.com/prod/variables/guest-search"
)
DATA_FILE = "gss.parquet"
VARIABLES_FILE = "variables.parquet"
QUESTIONS_FILE = "questions.parquet"
WEIGHT = "wtssps"
QUESTIONS_PER_PAGE = 100
# the suffix of the column holding each numeric column's extended missing values
MISSING_SUFFIX = "_null"
# how the stata metadata keys the labels of extended missing values
STATA_MISSING_PREFIX = "MISSING_"

# the fields kept of each variable the search api returns
QUESTIONS_SCHEMA = pl.Schema(
    {
        "name": pl.String,
        "description": pl.String,
        "surveyQuestion": pl.List(pl.String),
        "years": pl.List(
            pl.Struct(
                {
                    "year": pl.Int64,
                    "ballotInfo": pl.List(pl.String),
                    "isQuestionAvailable": pl.String,
                }
            )
        ),
        "module": pl.List(pl.String),
        "subject": pl.List(pl.String),
    }
)


def _in(column: str, low: int, high: int) -> pl.Expr:
    return pl.col(column).is_between(low, high)


DEMOGRAPHICS: dict[str, pl.Expr] = {
    # https://gssdataexplorer.norc.org/variables/81/vshow
    "sex": pl.col("sex").replace_strict(
        {1: "Male", 2: "Female"},
        default=None,
        return_dtype=schemas.SEX,
    ),
    # https://gssdataexplorer.norc.org/variables/82/vshow
    "race": pl.col("race").replace_strict(
        {1: "White", 2: "Black", 3: "Other"},
        default=None,
        return_dtype=schemas.RACE,
    ),
    # https://gssdataexplorer.norc.org/variables/287/vshow
    # codes without a category of their own (e.g. 6, buddhism) are "Other"
    "religion": pl.when(pl.col("relig").is_not_null()).then(
        pl.col("relig").replace_strict(
            {
                1: "Protestant",
                2: "Catholic",
                3: "Jewish",
                4: "None",
                5: "Other",
                11: "Protestant",
            },
            default=pl.lit("Other"),
            return_dtype=schemas.RELIGION,
        )
    ),
    # https://gssdataexplorer.norc.org/variables/53/vshow
    # age bin cuts taken from Pew:
    # https://www.pewresearch.org/politics/fact-sheet/party-affiliation-fact-sheet-npors/
    "age_bucket": (
        pl.col("age")
        .cut(
            breaks=[30, 50, 65],
            labels=schemas.AGE_BUCKET.categories.to_list(),
            left_closed=True,
        )
        .cast(schemas.AGE_BUCKET)
    ),
    # https://gssdataexplorer.norc.org/variables/141/vshow
    "political_party": (
        pl.when(_in("partyid", 0, 1))
        .then(pl.lit("Democrat"))
        .when(_in("partyid", 2, 4))
        .then(pl.lit("Independent"))
        .when(_in("partyid", 5, 6))
        .then(pl.lit("Republican"))
        .cast(schemas.POLITICAL_PARTY)
    ),
    # https://gssdataexplorer.norc.org/variables/178/vshow
    "political_ideology": (
        pl.when(_in("polviews", 1, 3))
        .then(pl.lit("Liberal"))
        .when(_in("polviews", 5, 7))
        .then(pl.lit("Conservative"))
        .when(pl.col("polviews") == 4)
        .then(pl.lit("Moderate"))
        .cast(schemas.POLITICAL_IDEOLOGY)
    ),
    # https://gssdataexplorer.norc.org/variables/3/vshow
    "workforce_status": (
        pl.when(_in("wrkstat", 1, 3))
        .then(pl.lit("Employed"))
        .when(pl.col("wrkstat") == 4)
        .then(pl.lit("Unemployed"))
        .when(_in("wrkstat", 5, 8))
        .then(pl.lit("Out of labor force"))
        .cast(schemas.WORKFORCE_STATUS)
    ),
    # https://gssdataexplorer.norc.org/variables/59/vshow
    # `degree` codes 3 and 4 are bachelor's and graduate degrees
    "education": (
        pl.when(pl.col("degree") < 3)
        .then(pl.lit("No Bachelor's degree"))
        .when(pl.col("degree") >= 3)
        .then(pl.lit("Bachelor's degree or more"))
        .cast(schemas.EDUCATION)
    ),
    # https://gssdataexplorer.norc.org/variables/94/vshow
    # https://gssdataexplorer.norc.org/variables/95/vshow
    "self_or_parent_immigrant": (
        # self born elsewhere
        pl.col("born").eq(2)
        # or either parent born elsewhere
        | pl.col("parborn").is_in([1, 2, 4, 6, 8])
    ),
    # https://gssdataexplorer.norc.org/variables/119/vshow
    "geographic_region": pl.col("region").replace_strict(
        {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
        default=None,
        return_dtype=schemas.GEOGRAPHIC_REGION,
    ),
    # https://gssdataexplorer.norc.org/variables/120/vshow
    "type_of_neighborhood": (
        pl.when(_in("xnorcsiz", 1, 2))
        .then(pl.lit("Urban"))
        .when(_in("xnorcsiz", 3, 6))
        .then(pl.lit("Suburban"))
        .when(_in("xnorcsiz", 7, 10))
        .then(pl.lit("Rural"))
        .cast(schemas.TYPE_OF_NEIGHBORHOOD)
    ),
}

# the stata file stores these as small integers
YEAR = pl.col("year").cast(pl.Int64)
RESPONDENT = pl.col("id").cast(pl.Int64).alias("respondent")


VARIABLES_SCHEMA = pl.Schema(
    {
        "variable": pl.String,
        "description": pl.String,
        "values": pl.List(pl.Int64),
        "labels": pl.List(pl.String),
        "missing_codes": pl.List(pl.String),
        "missing_labels": pl.List(pl.String),
    }
)


def missing_code(label_key: str) -> str | None:
    """
    The extended missing value a value label's key names, e.g. `MISSING_d` -> `.d`.

    Null for keys of ordinary (integer) values.
    """
    letter = label_key.removeprefix(STATA_MISSING_PREFIX)
    return f".{letter}" if letter != label_key else None


def variables_table(metadata: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """
    One row per variable: its description, value labels, and missing-code labels.

    Missing codes are spelled as in the data's `_null` columns (e.g. `.d`).
    """

    def variable_row(variable: Mapping[str, Any]) -> dict[str, Any]:
        labels = variable.get("value_labels") or {}
        values = sorted((int(k), v) for k, v in labels.items() if not missing_code(k))
        missing = sorted(
            (code, v) for k, v in labels.items() if (code := missing_code(k))
        )

        return {
            "variable": variable["name"],
            "description": variable.get("label"),
            "values": [k for k, _ in values],
            "labels": [v for _, v in values],
            "missing_codes": [k for k, _ in missing],
            "missing_labels": [v for _, v in missing],
        }

    return pl.DataFrame(map(variable_row, metadata), schema=VARIABLES_SCHEMA)


def write_raw(dta: Path, dest: Path) -> None:
    """
    Write the GSS data and its variable metadata as parquet.

    Each numeric column `x` is followed by a string column `x_null`
    holding the extended missing value (e.g. `.d`) where `x` is null.
    Strings are decoded as utf-8, which stata 14+ files (format 118) always use.
    """
    reader = prs.ScanReadstat(
        str(dta),
        preserve_order=True,
        informative_nulls=prs.InformativeNullOpts(
            suffix=MISSING_SUFFIX,
            use_value_labels=False,
        ),
    )
    reader.df.sink_parquet(dest / DATA_FILE)
    variables_table(reader.metadata["variables"]).write_parquet(dest / VARIABLES_FILE)


def search_variables(client: httpx.Client, page: int, limit: int) -> dict:
    """One page of the explorer's variable search, over every year and subject."""
    request = {
        "page": page,
        "limit": limit,
        "subjects": None,
        "modules": None,
        "my_tags": None,
        "shared_tags": None,
        # the api requires a range; this one covers every year
        "yearRange": [1972, 2100],
        "years": None,
        "parameter": None,
        "term": None,
        "workspaceId": None,
        "yearBallot": None,
    }
    response = client.post(QUESTIONS_URL, json=request)
    response.raise_for_status()
    return response.json()


def download_questions(dest: Path) -> None:
    """Write every variable's question text, as the data explorer publishes it."""
    with httpx.Client(timeout=60) as client:
        total = search_variables(client, page=1, limit=1)["totalVarCount"]
        pages = range(1, math.ceil(total / QUESTIONS_PER_PAGE) + 1)
        variables = [
            variable
            for page in pages
            for variable in search_variables(client, page, QUESTIONS_PER_PAGE)[
                "variables"
            ]
        ]

    pl.DataFrame(
        [{k: v.get(k) for k in QUESTIONS_SCHEMA} for v in variables],
        schema=QUESTIONS_SCHEMA,
    ).write_parquet(dest / QUESTIONS_FILE)


def question_text(questions: pl.LazyFrame) -> pl.LazyFrame:
    """
    The text of each variable's question, keyed like the data's columns.

    The explorer names a few variables differently from the stata file:
    in upper or mixed case, or with a trailing underscore (`id_`, `class_`).
    """
    return questions.select(
        pl.col("name").str.to_lowercase().str.strip_suffix("_").alias("variable"),
        pl.col("surveyQuestion").list.first().alias("question"),
    ).unique("variable", keep="first", maintain_order=True)


def years_observed(gss: pl.LazyFrame) -> pl.LazyFrame:
    """The sorted years in which each variable has any non-null value."""
    return (
        gss.group_by(YEAR.alias("__year"))
        .agg(cs.exclude(cs.ends_with(MISSING_SUFFIX)).is_not_null().any())
        .unpivot(index="__year")
        .filter(pl.col("value"))
        .group_by("variable")
        .agg(pl.col("__year").sort().alias("years"))
    )


def codebook(
    gss: pl.LazyFrame,
    variables: pl.LazyFrame,
    questions: pl.LazyFrame,
) -> pl.LazyFrame:
    def by_variable(left: pl.LazyFrame, right: pl.LazyFrame) -> pl.LazyFrame:
        return left.join(
            right,
            on="variable",
            how="left",
            validate="1:1",
            maintain_order="left",
        )

    return (
        variables.pipe(by_variable, years_observed(gss))
        .pipe(by_variable, question_text(questions))
        .select(
            pl.col("variable").cast(pl.Categorical),
            "description",
            "question",
            pl.col("years").fill_null([]),
            pl.struct(value="values", label="labels").alias("labels"),
        )
    )


def respondents(gss: pl.LazyFrame) -> pl.LazyFrame:
    return gss.select(
        YEAR,
        RESPONDENT,
        pl.col(WEIGHT).alias("weight"),
        **DEMOGRAPHICS,
    )


def responses(gss: pl.LazyFrame, variables: Sequence[str]) -> pl.LazyFrame:
    """
    Answers to `variables` in long format.

    Missing codes are stata extended missing values, already null in the raw data.
    """
    return (
        gss.select(YEAR, RESPONDENT, *variables)
        .unpivot(index=["year", "respondent"])
        .drop_nulls("value")
        .with_columns(
            pl.col("variable").cast(pl.Categorical),
            pl.col("value").cast(pl.Int64),
        )
    )


class GSS(Source):
    """The cumulative GSS file, 1972 onward."""

    name = "gss"
    demographics = frozenset(DEMOGRAPHICS)

    def __init__(
        self,
        root: Path = DATA_ROOT,
        variables: Sequence[str] | None = None,
    ) -> None:
        super().__init__(root)
        self.variables = list(
            question_variables(self.name) if variables is None else variables
        )

    @override
    def download(self, dest: Path) -> None:
        with tempfile.TemporaryDirectory(dir=dest) as scratch:
            archive = Path(scratch) / "GSS_stata.zip"

            with httpx.stream("GET", URL, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                with archive.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)

            with ZipFile(archive) as z:
                dta = z.extract(
                    next(f for f in z.namelist() if f.endswith(".dta")),
                    scratch,
                )

            write_raw(Path(dta), dest)

        download_questions(dest)

    @override
    def clean(self, raw: Path) -> CleanFrames:
        variables = pl.scan_parquet(raw / VARIABLES_FILE)

        if "missing_codes" not in variables.collect_schema():
            raise ValueError(
                f"The raw GSS data in {raw} predates missing codes; "
                "refetch it with `GSS().fetch(refresh=True)`"
            )

        gss = pl.scan_parquet(raw / DATA_FILE)

        return CleanFrames(
            codebook=codebook(
                gss,
                variables,
                pl.scan_parquet(raw / QUESTIONS_FILE),
            ),
            respondents=respondents(gss),
            responses=responses(gss, self.variables),
        )
