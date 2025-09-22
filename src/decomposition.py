import polars as pl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from rich.progress import track

from src.data_prep import load_gss_data
from src.variables import subset_to_study
from src.mca import mca


def main():
    responses, meta = load_gss_data(use_cache=True)

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
    pred = pl.col.year.eq(2018)
    for year, _, eigenvalues in mca_no_correlation.filter(pred).iter_rows():
        plt.plot(eigenvalues[:n_components], alpha=0.15)
    for year, _, eigenvalues in mca_bootstrap.filter(pred).iter_rows():
        plt.plot(eigenvalues[:n_components], alpha=0.15)
    plt.plot(
        mca_results.filter(year=year).get_column("eigenvalues").first()[:n_components],
        color="red",
    )
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
