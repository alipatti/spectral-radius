"""
Taken from https://gssdataexplorer.norc.org/variables/vfilter
"""

import polars as pl


DEMOGRAPHIC_VARIABLES: dict[str, pl.Expr] = dict(
    # https://gssdataexplorer.norc.org/variables/81/vshow
    sex=pl.col("sex").replace_strict(
        {1: "Male", 2: "Female"},
        return_dtype=pl.Categorical,
        default=None,
    ),
    #
    race=pl.col("race").replace_strict(
        {1: "White", 2: "Black", 3: "Other"},
        return_dtype=pl.Categorical,
        default=None,
    ),
    # https://gssdataexplorer.norc.org/variables/53/vshow
    # age bin cuts taken from Pew:
    # https://www.pewresearch.org/politics/fact-sheet/party-affiliation-fact-sheet-npors/
    age_bucket=(
        pl.col("age")
        .cut(
            breaks=[30, 50, 65],
            labels=["18-29", "30-49", "50-65", "65+"],
            left_closed=True,
        )
        .cast(pl.Categorical())
    ),
    # https://gssdataexplorer.norc.org/variables/141/vshow
    political_party=(
        pl.when(pl.col("partyid").is_between(0, 1))
        .then(pl.lit("Democrat"))
        .when(pl.col("partyid").is_between(2, 4))
        .then(pl.lit("Independent"))
        .when(pl.col("partyid").is_between(5, 6))
        .then(pl.lit("Republican"))
        .cast(pl.Categorical)
    ),
    # https://gssdataexplorer.norc.org/variables/178/vshow
    political_ideology=(
        pl.when(pl.col("polviews").is_between(1, 3))
        .then(pl.lit("Liberal"))
        .when(pl.col("polviews").is_between(5, 7))
        .then(pl.lit("Conservative"))
        .when(pl.col("polviews") == 4)
        .then(pl.lit("Moderate"))
        .cast(pl.Categorical)
    ),
    # https://gssdataexplorer.norc.org/variables/3/vshow
    workforce_status=(
        pl.when(pl.col("wrkstat").is_between(1, 3))
        .then(pl.lit("Employed"))
        .when(pl.col("wrkstat") == 4)
        .then(pl.lit("Unmployed"))
        .when(pl.col("wrkstat").is_between(5, 8))
        .then(pl.lit("Out of labor force"))
        .cast(pl.Categorical)
    ),
    # https://gssdataexplorer.norc.org/variables/59/vshow
    education=(
        pl.when(pl.col("degree") < 3)
        .then(pl.lit("No Bachelor's degree"))
        .when(pl.col("educ") >= 3)
        .then(pl.lit("Bachelor's degree or more"))
        .cast(pl.Categorical)
    ),
    # https://gssdataexplorer.norc.org/variables/94/vshow
    # https://gssdataexplorer.norc.org/variables/95/vshow
    self_or_parent_immigrant=(
        # self born elsewhere
        pl.col("born").eq(2)
        # OR either parent born elsewhere
        | pl.col("parborn").is_in([1, 2, 4, 6, 8])
    ),
    # https://gssdataexplorer.norc.org/variables/119/vshow
    region=pl.col("region").replace_strict(
        {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
        return_dtype=pl.Categorical,
    ),
    # https://gssdataexplorer.norc.org/variables/120/vshow
    place_type=(
        pl.when(pl.col("xnorcsiz").is_between(1, 2))
        .then(pl.lit("Urban"))
        .when(pl.col("xnorcsiz").is_between(3, 6))
        .then(pl.lit("Suburban"))
        .when(pl.col("xnorcsiz").is_between(7, 10))
        .then(pl.lit("Rural"))
        .cast(pl.Categorical)
    ),
)
