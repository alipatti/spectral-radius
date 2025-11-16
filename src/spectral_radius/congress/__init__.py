from pathlib import Path

import requests
import polars as pl


def get_roll_call_data(
    issues_url="https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.csv",
    votes_url="https://voteview.com/static/data/out/votes/HSall_votes.csv",
    cache_path=Path("data/congress/votes.parquet"),
):
    if not cache_path.exists():
        print(f"Downloading {votes_url} to {cache_path}")

        with requests.get(votes_url) as votes:
            print("Loading vote data")
            raw_votes = pl.read_csv(
                "~/Downloads/HSall_votes.csv",  # TODO: replace with votes.content
                null_values=["N/A"],
                infer_schema_length=None,
            ).lazy()

        with requests.get(issues_url) as issues:
            print("Loading issue data")
            raw_issues = pl.read_csv(
                "~/Downloads/HSall_rollcalls.csv",  # TODO: replace with issues.content
                null_values=["N/A"],
                infer_schema_length=None,
            ).lazy()

        # columns that identify each vote
        id_cols = ["congress", pl.col("chamber").cast(pl.Categorical()), "rollnumber"]

        # see https://voteview.com/articles/data_help_votes
        clean_votes = raw_votes.select(
            *id_cols,
            voter_id="icpsr",
            vote_cast=pl.col("cast_code")  # yea or nay
            .cast(pl.UInt8)
            .pipe(
                lambda x: pl.when(x.is_between(1, 3))
                .then(1)  # yea
                .when(x.is_between(4, 6))
                .then(0)  # nay
            ),
        )

        clean_issues = raw_issues.select(
            *id_cols,
            date=pl.col("date").str.to_date(),
            bill="bill_number",
            question="vote_question",
            description="vote_desc",
        )

        clean = clean_issues.join(
            clean_votes,
            on=id_cols,
            validate="m:1",
        )

        print(clean)
        print(clean.schema)

        clean.sink_parquet(cache_path)

        print(" - Done!")

    return pl.read_parquet(cache_path)


if __name__ == "__main__":
    get_roll_call_data()
