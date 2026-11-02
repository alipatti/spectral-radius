"""
The General Social Survey.

Codes are documented at https://gssdataexplorer.norc.org/variables/vfilter.
"""

import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, override
from zipfile import ZipFile

import httpx
import polars as pl
import polars.selectors as cs
import polars_readstat as prs

from spectral_radius.surveys import scales, schemas
from spectral_radius.surveys.question_sets import question_variables
from spectral_radius.surveys.scales import Scale
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

# the kind of each gss missing code (a stata extended missing value),
# after the label the gss publishes for it, quoted where the kind differs
MISSING_CODES: dict[str, str] = {
    ".d": "dont_know",  # also "can't choose" for some variables
    ".f": "other",  # "missing birthdate information", for one variable
    ".i": "inapplicable",  # "iap": not asked, e.g. by a filter or ballot
    ".j": "inapplicable",  # "I don't have a job"
    ".m": "unspecified",  # "dk, na, iap"
    ".n": "no_answer",
    ".p": "not_imputable",
    ".r": "refused",
    ".s": "skipped",  # "skipped on web"
    ".u": "uncodeable",
    ".x": "not_released",  # "not available in this release"
    ".y": "not_asked_this_year",  # "not available in this year"
    ".z": "other",  # "see codebook"
}


def missing_kind(code: pl.Expr) -> pl.Expr:
    """The `MISSING` kind of each gss missing code, failing on an unknown code."""
    return code.replace_strict(MISSING_CODES, return_dtype=schemas.MISSING)


# the scale of each opinion variable the gss can read, checked against the
# gss's value labels on load; a variable must be assigned one here to be read
VARIABLE_SCALES: Mapping[str, Scale] = MappingProxyType(
    {
        "spkath": scales.ALLOWED_TO_SPEAK,
        "spkmil": scales.ALLOWED_TO_SPEAK,
        "spkrac": scales.ALLOWED_TO_SPEAK,
        "spkhomo": scales.ALLOWED_TO_SPEAK,
        "spkmslm": scales.ALLOWED_TO_SPEAK,
        "libath": scales.REMOVE_BOOK,
        "libmil": scales.REMOVE_BOOK,
        "librac": scales.REMOVE_BOOK,
        "libhomo": scales.REMOVE_BOOK,
        "libmslm": scales.REMOVE_BOOK,
        "colath": scales.ALLOWED_TO_TEACH,
        "colmil": scales.ALLOWED_TO_TEACH,
        "colrac": scales.ALLOWED_TO_TEACH,
        "colhomo": scales.ALLOWED_TO_TEACH,
        "colmslm": scales.ALLOWED_TO_TEACH,
        "advfront": scales.AGREE_4,
        "natspac": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natenvir": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natheal": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natcity": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natcrime": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natdrug": scales.TOO_LITTLE_TO_TOO_MUCH,
        "nateduc": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natrace": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natarms": scales.TOO_LITTLE_TO_TOO_MUCH,
        "nataid": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natfare": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natroad": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natsoc": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natmass": scales.TOO_LITTLE_TO_TOO_MUCH,
        "natpark": scales.TOO_LITTLE_TO_TOO_MUCH,
        "helpblk": scales.GOVERNMENT_TO_PEOPLE,
        "helpsick": scales.GOVERNMENT_TO_PEOPLE,
        "helppoor": scales.GOVERNMENT_TO_PEOPLE,
        "affrmact": scales.FAVOR_OPPOSE_4,
        "fejobaff": scales.FAVOR_OPPOSE_4,
        "fehire": scales.AGREE_5,
        "wrkwayup": scales.AGREE_5,
        "racdif1": scales.YES_NO,
        "racdif2": scales.YES_NO,
        "racdif3": scales.YES_NO,
        "racdif4": scales.YES_NO,
        "eqwlth": scales.REDUCE_INCOME_DIFFERENCES,
        "polhitok": scales.YES_NO,
        "polabuse": scales.YES_NO,
        "polmurdr": scales.YES_NO,
        "polescap": scales.YES_NO,
        "polattak": scales.YES_NO,
        "courts": scales.COURTS_HARSHNESS,
        "cappun": scales.FAVOR_OPPOSE,
        "conjudge": scales.CONFIDENCE,
        "premarsx": scales.ALWAYS_TO_NOT_WRONG,
        "xmarsex": scales.ALWAYS_TO_NOT_WRONG,
        "homosex": scales.ALWAYS_TO_NOT_WRONG,
        "teensex": scales.ALWAYS_TO_NOT_WRONG,
        "sexeduc": scales.FAVOR_OPPOSE,
        "pillok": scales.AGREE_4,
        "abdefect": scales.YES_NO,
        "abnomore": scales.YES_NO,
        "abhlth": scales.YES_NO,
        "abpoor": scales.YES_NO,
        "abrape": scales.YES_NO,
        "absingle": scales.YES_NO,
        "abany": scales.YES_NO,
    }
)
OFF_SCALE = pl.lit("off_scale", dtype=schemas.MISSING)


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
    read: Mapping[str, Scale],
) -> pl.LazyFrame:
    """
    One row per variable of the GSS.

    `read` gives the scale of each variable read into responses;
    the labeled off-scale codes of that scale are listed after the variable's
    missing codes, with kind `off_scale`.
    """

    def by_variable(left: pl.LazyFrame, right: pl.LazyFrame) -> pl.LazyFrame:
        return left.join(
            right,
            on="variable",
            how="left",
            validate="1:1",
            maintain_order="left",
        )

    def with_off_scale(missing: pl.Expr, off: str) -> pl.Expr:
        return pl.concat_list(missing, pl.col(off).fill_null([]))

    return (
        variables.pipe(by_variable, years_observed(gss))
        .pipe(by_variable, question_text(questions))
        .pipe(by_variable, off_scale_labels(variables, read))
        .select(
            pl.col("variable").cast(pl.Categorical),
            "description",
            "question",
            pl.col("years").fill_null([]),
            pl.struct(value="values", label="labels").alias("labels"),
            pl.struct(
                code=with_off_scale(pl.col("missing_codes"), "off_scale_codes"),
                label=with_off_scale(pl.col("missing_labels"), "off_scale_labels"),
                kind=with_off_scale(
                    pl.col("missing_codes").list.eval(missing_kind(pl.element())),
                    "off_scale_kinds",
                ),
            ).alias("missing"),
            *schemas.read_columns(pl.col("variable"), read),
        )
    )


def off_scale_labels(
    variables: pl.LazyFrame,
    read: Mapping[str, Scale],
) -> pl.LazyFrame:
    """
    The off-scale codes (as strings), labels, and kinds each variable labels.

    A variable's off-scale codes are those its scale declares.
    """
    codes = pl.LazyFrame(
        [
            {"variable": variable, "value": code}
            for variable, scale in read.items()
            for code in scale.off_scale_codes
        ],
        schema={"variable": pl.String, "value": pl.Int64},
    )

    return (
        variables.select("variable", value="values", label="labels", kind=OFF_SCALE)
        .explode("value", "label", empty_as_null=False)
        .join(codes, on=["variable", "value"], how="semi")
        .group_by("variable", maintain_order=True)
        .agg(
            pl.col("value").cast(pl.String).alias("off_scale_codes"),
            pl.col("label").alias("off_scale_labels"),
            pl.col("kind").alias("off_scale_kinds"),
        )
    )


def respondents(gss: pl.LazyFrame) -> pl.LazyFrame:
    return gss.select(
        YEAR,
        RESPONDENT,
        pl.col(WEIGHT).alias("weight"),
        **DEMOGRAPHICS,
    )


def response(variable: str, scale: Scale) -> pl.Expr:
    """
    A variable's value and the kind of each non-answer, as a struct.

    The raw code is kept as the value, except that a null takes its kind
    from its missing code (or is `unrecorded` if it has none)
    and an off-scale code of `scale` is moved out of `value` with kind `off_scale`.
    Any other code is kept, and fails validation unless it's a code of `scale`.
    """
    raw = pl.col(variable).cast(pl.Int64)
    is_off_scale = raw.is_in(scale.off_scale_codes)
    unrecorded = pl.lit(schemas.UNRECORDED, dtype=schemas.MISSING)
    code = pl.col(f"{variable}{MISSING_SUFFIX}")

    return pl.struct(
        value=pl.when(is_off_scale.not_()).then(raw),
        missing=pl.when(is_off_scale)
        .then(OFF_SCALE)
        .when(raw.is_null())
        .then(missing_kind(code).fill_null(unrecorded)),
    ).alias(variable)


def responses(gss: pl.LazyFrame, read: Mapping[str, Scale]) -> pl.LazyFrame:
    """
    Every respondent's response to each variable, in long format.

    `read` maps each variable to read to its scale.
    """
    return (
        gss.select(YEAR, RESPONDENT, *(response(v, s) for v, s in read.items()))
        .unpivot(index=["year", "respondent"], value_name="response")
        .select(
            "year",
            "respondent",
            pl.col("variable").cast(pl.Categorical),
            pl.col("response").struct.unnest(),
        )
    )


class GSS(Source):
    """The cumulative GSS file, 1972 onward."""

    name = "gss"
    demographics = frozenset(DEMOGRAPHICS)
    start_year = 1990

    def __init__(
        self,
        root: Path = DATA_ROOT,
        variables: Sequence[str] | None = None,
    ) -> None:
        super().__init__(root)
        self.variables = list(
            question_variables(self.name) if variables is None else variables
        )

        if unassigned := [v for v in self.variables if v not in VARIABLE_SCALES]:
            raise ValueError(f"GSS variables without an assigned scale: {unassigned}")

    @property
    @override
    def scales(self) -> Mapping[str, Scale]:
        return MappingProxyType({v: VARIABLE_SCALES[v] for v in self.variables})

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
                self.scales,
            ),
            respondents=respondents(gss),
            responses=responses(gss, self.scales),
        )
