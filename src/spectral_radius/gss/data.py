from io import BytesIO
from typing import Any
from zipfile import ZipFile
import requests
import polars as pl
import tempfile
from itertools import chain
import math
from rich.progress import track
import pyreadstat
from pathlib import Path
import pickle
from rich import print
import inflection


def load_gss_data(
    clean_data_dir=Path("data/gss"),
    use_cache=True,
    url="https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip",
) -> tuple[pl.DataFrame, pyreadstat.metadata_container]:
    """http://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip"""

    metadata_path = clean_data_dir / "metadata.pickle"
    parquet_path = clean_data_dir / "gss.parquet"

    if not clean_data_dir.exists() or not use_cache:
        print("[bold][blue]Parquet GSS data not found. Creating it.")

        print(" - Downloading from gss.norc.org...")
        with requests.get(url) as r:
            z = ZipFile(BytesIO(r.content))

            dta_path = z.extract(
                next(f for f in z.filelist if f.filename.endswith(".dta")),
                tempfile.mkdtemp(),
            )

        print(" - Converting Stata dta file to parquet...")
        df, metadata = pyreadstat.read_dta(
            dta_path,
            encoding="LATIN1",
        )

        clean_data_dir.mkdir(parents=True, exist_ok=True)
        print(" - Saving parquet...")
        pl.from_pandas(df).write_parquet(parquet_path)

        print(" - Saving metadata...")
        metadata_path.write_bytes(pickle.dumps(metadata))

    return pl.read_parquet(parquet_path), pickle.loads(metadata_path.read_bytes())


# TODO: do we want to use this?
def label_gss_variables(
    df: pl.DataFrame,
    metadata: pyreadstat.metadata_container,
    cols_to_label: list[str],
) -> pl.DataFrame:
    def _labels_to_list(colname, mapping):
        ints = [k for k in mapping.keys() if isinstance(k, int)]
        print(colname, max(ints, default=0), ints)

        return [mapping.get(i) for i in range(1, max(ints, default=0) + 1)]

    labels = {
        col: _labels_to_list(col, mapping)
        for col, mapping in metadata.variable_value_labels.items()
    }

    return df.with_columns(
        pl.col(col).replace_strict(
            labels[col],
            return_dtype=pl.Categorical(ordering="lexical"),
        )
        for col in cols_to_label
    )


def get_all_gss_variables(
    clean_parquet_file=Path("data/gss/variables.parquet"),
    api_url=(
        "https://3ilfsaj2lj.execute-api.us-east-1.amazonaws.com/prod/variables/guest-search"
    ),
    default_request=dict(
        page=1,
        limit=25,
        subjects=None,
        modules=None,
        my_tags=None,
        shared_tags=None,
        yearRange=[1972, 2022],
        years=None,
        parameter=None,
        term=None,
        workspaceId=None,
        yearBallot=None,
    ),
    use_cache=True,
) -> pl.DataFrame:
    def _make_gss_api_request(**kwargs) -> dict[str, Any]:
        return requests.post(api_url, json=default_request | kwargs).json()

    if not clean_parquet_file.exists() or not use_cache:
        total_variables = _make_gss_api_request()["totalVarCount"]
        vars_per_page = 100
        n_pages = math.ceil(total_variables / vars_per_page)

        responses = [
            _make_gss_api_request(page=i, limit=vars_per_page)["variables"]
            for i in track(
                range(1, n_pages + 1),
                description="Getting variable descriptions from the GSS website...",
            )
        ]

        (
            pl.json_normalize(list(chain.from_iterable(responses)))
            # .explode("years")
            # .unnest("years")
            .rename(inflection.underscore)
            .with_columns(
                pl.col.survey_question.list.first(),
                pl.col("module", "subject").cast(
                    pl.List(pl.Categorical(ordering="lexical"))
                ),
                # pl.col("is_question_available").cast(
                #     pl.Categorical(ordering="lexical")
                # ),
            )
            .drop("tag_info")
            .write_parquet(clean_parquet_file)
        )

    return pl.read_parquet(clean_parquet_file)
