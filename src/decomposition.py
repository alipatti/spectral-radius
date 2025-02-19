from typing import Literal

import polars as pl

import numpy as np
from numpy.linalg import svd
from numpy.linalg.linalg import SVDResult
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from rich.progress import track

from src.data_prep import load_gss_data
from src.variables import subset_to_study


def mca(df: pl.DataFrame) -> SVDResult:
    # https://en.wikipedia.org/wiki/Multiple_correspondence_analysis#Details

    X = df.drop("year", "sample", strict=False).to_dummies().to_numpy()

    Z = X / X.sum()
    r = Z.sum(axis=1)
    c = Z.sum(axis=0)

    M = np.diag(1 / np.sqrt(r)) @ (Z - np.outer(r, c)) @ np.diag(1 / np.sqrt(c))

    return svd(M)


def _sample_from_cdf(cdf: pl.Series, n) -> pl.Series:
    return cdf.search_sorted(np.random.random_sample(n))


def simulate_data(
    true_data: pl.DataFrame,
    resampling_method: Literal["marginal", "bootstrap"] = "marginal",
    resamples=1,
    seed: int = 12413528,
) -> pl.DataFrame:
    np.random.seed(seed)

    if resampling_method == "marginal":
        cdfs = (
            true_data.cast({pl.Categorical: pl.String})
            .unpivot(index="year")
            .sort(pl.selectors.all())
            .group_by(pl.selectors.all(), maintain_order=True)
            .len("count")
            .with_columns(
                (pl.col("count") / pl.col("count").sum())  # pdf
                .cum_sum()  # cdf
                .over("year", "variable")
                .alias("cdf")
            )
            .partition_by("year", "variable", as_dict=True)
        )

        # sorry this is heinous ...
        return (
            pl.concat(
                pl.DataFrame(
                    dict(
                        year=year,
                        sample=range(1, resamples + 1),
                        column=column,
                        values=[
                            df["value"][_sample_from_cdf(df["cdf"], df["count"].sum())]
                            for _ in range(resamples)
                        ],
                    )
                )
                for (year, column), df in cdfs.items()
            )
            .pivot(on="column", index=["year", "sample"], values="values")
            .explode(pl.selectors.exclude("year", "sample"))
        )

    elif resampling_method == "bootstrap":
        return pl.concat(
            df.sample(
                fraction=1,
                with_replacement=True,
                seed=np.random.randint(1, 1_000_000),
                shuffle=True,
            ).with_columns(pl.lit(sample).alias("sample"))
            for sample in range(1, resamples + 1)
            for df in true_data.partition_by("year")
        )

    else:
        raise NotImplementedError


responses, meta = load_gss_data()

variables = (
    # subset_to_study["government_spending"]
    subset_to_study["freedom_of_speech"]
    + subset_to_study["reproductive_rights"]
)

subset = (
    responses.select("year", *variables)
    # .pipe(label_gss_variables, meta, variables)
    .filter(pl.col("year").ge(2000)).drop_nulls()
)


mca_results = pl.from_records(
    [
        (df.get_column("year").first(), *mca(df))
        for df in track(
            subset.partition_by("year"),
            description="Running MCA...",
        )
    ],
    orient="row",
    schema=["year", "u", "eigenvalues", "vT"],
).select(
    "year",
    pl.col("eigenvalues").map_elements(
        lambda s: pl.Series(s**2),
    ),
)

d = len(mca_results["eigenvalues"].first())  # type: ignore

plt.clf()
n_components = 18
for year, eigenvalues in mca_results.select(
    (pl.col.year - pl.col.year.min()) / (pl.col.year.max() - pl.col.year.min()),
    "eigenvalues",
).iter_rows():
    plt.plot(eigenvalues[:n_components], color=cm.viridis(year))
plt.plot([], [])
plt.yscale("log")
plt.show(block=False)


# independence counterfactual
mca_no_correlation = pl.from_records(
    [
        (df.get_column("year").first(), df.get_column("sample").first(), *mca(df))
        for df in track(
            simulate_data(subset, resamples=100).partition_by("year", "sample"),
            description="Running MCA...",
        )
    ],
    orient="row",
    schema=["year", "sample", "u", "eigenvalues", "vT"],
).select(
    "year",
    "sample",
    pl.col("eigenvalues").map_elements(lambda s: pl.Series(s**2)),
)

# independence counterfactual
mca_bootstrap = pl.from_records(
    [
        (df.get_column("year").first(), df.get_column("sample").first(), *mca(df))
        for df in track(
            simulate_data(
                subset, resamples=100, resampling_method="bootstrap"
            ).partition_by("year", "sample"),
            description="Running MCA...",
        )
    ],
    orient="row",
    schema=["year", "sample", "u", "eigenvalues", "vT"],
).select(
    "year",
    "sample",
    pl.col("eigenvalues").map_elements(lambda s: pl.Series(s**2)),
)


plt.clf()
n_components = 18
for year, sample, eigenvalues in mca_no_correlation.filter(
    pl.col.year.is_in([2002, 2018, 2010])
).iter_rows():
    plt.plot(eigenvalues[:n_components], alpha=0.15)
# plt.plot(
#     mca_results.filter(year=year)["eigenvalues"].first()[:n_components], color="red"
# )
plt.yscale("log")
plt.show(block=False)

mca_no_correlation.with_columns(index=pl.int_range(0, d).implode()).explode(
    "index", "eigenvalues"
).group_by("year", "index").agg(
    pl.col("eigenvalues").mean().alias("mean"),
    pl.col("eigenvalues").std().alias("std"),
).sort(
    pl.selectors.all()
).join(
    mca_results.with_columns(index=pl.int_range(0, d).implode()).explode(
        "index", "eigenvalues"
    ),
    on=["year", "index"],
).with_columns(
    significance_ratio=pl.col("eigenvalues") / pl.col("mean")
).filter(
    pl.col("index") < 15
).plot.line(
    x="index",
    y="significance_ratio",
    color="year:N",
).save(
    "/Users/ali/Downloads/eigenplot.pdf"
)
