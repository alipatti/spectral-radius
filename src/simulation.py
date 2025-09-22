from typing import Literal

import numpy as np
import polars as pl


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
        # assumes independence and resamples from the marginals
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
                # resample a new seed each time so that we get different draws
                seed=np.random.randint(1, 10_000_000),
                shuffle=True,
            ).with_columns(pl.lit(sample).alias("sample"))
            for sample in range(1, resamples + 1)
            for df in true_data.partition_by("year")
        )

    else:
        raise NotImplementedError
