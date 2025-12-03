from functools import cache

import polars as pl

from spectral_radius.gss.demographic_variables import DEMOGRAPHIC_VARIABLES
from spectral_radius.gss.opinion_variables import OPINION_CATEGORIES, OPINION_VARIABLES
from spectral_radius.gss.data import load_gss_data
from spectral_radius.variable_coding import scale_to_pm_1
from spectral_radius.constants import START_YEAR


@cache
def get_gss(*, weight="wtssps", encoding_function=scale_to_pm_1):
    return (
        load_gss_data(use_cache=True)[0]
        .select(
            "year",
            # demographic variables
            *(v.alias(k) for k, v in DEMOGRAPHIC_VARIABLES.items()),
            # weight
            pl.col(weight).alias("w"),
            # analysis variables
            pl.col(OPINION_VARIABLES).pipe(encoding_function),
        )
        .filter(pl.col("year") >= START_YEAR)
    )


__all__ = [
    "get_gss",
    "DEMOGRAPHIC_VARIABLES",
    "OPINION_VARIABLES",
    "OPINION_CATEGORIES",
]
