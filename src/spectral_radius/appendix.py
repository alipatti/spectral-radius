"""The paper's appendix listing each question set's questions, from the codebook."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import inflection
import polars as pl

from spectral_radius.analysis import survey_data
from spectral_radius.surveys import QUESTION_SETS

OUTPUT = Path("paper/generated")


def escape_tex(s: str) -> str:
    return s.replace("&", r"\&").replace("\n", r"\\").replace(". . .", r"$\ldots$ ")


def format_question(
    variable: str,
    description: str | None,
    question: str | None,
    responses: str,
) -> str:
    return rf"""
    \textsc{{{variable}}} \hfill \textit{{({escape_tex(inflection.humanize(description or ""))})}}
    \begin{{quote}}
        \footnotesize
        {escape_tex(question or "")}

        \textit{{Possible responses}}: {escape_tex(responses)}
    \end{{quote}}
    """


def questions_appendix(
    codebook: pl.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> str:
    """One subsection per category, listing its variables from `codebook` by name."""
    labels = pl.col("labels").struct
    labeled_values = pl.format(
        "``{}'' ({})",
        pl.col("label"),
        pl.col("value"),
    )

    responses = (
        codebook.select(
            pl.col("variable").cast(pl.String),
            "description",
            "question",
            labels.field("value"),
            labels.field("label"),
        )
        .explode("value", "label", empty_as_null=True)
        .group_by("variable", maintain_order=True)
        .agg(
            pl.first("description", "question"),
            labeled_values.drop_nulls().str.join(", ").alias("responses"),
        )
    )

    questions = (
        pl.DataFrame(
            [
                {"category": category, "variable": variable}
                for category, variables in categories.items()
                for variable in variables
            ]
        )
        .join(responses, on="variable", how="left", validate="m:1")
        .sort("category", "variable")
    )

    return "\n".join(
        rf"\subsection{{{category}}}"
        + "\n\n".join(format_question(*q) for q in group.iter_rows())
        for (category,), group in questions.partition_by(
            "category",
            maintain_order=True,
            as_dict=True,
            include_key=False,
        ).items()
    )


def main(survey: str = "gss") -> None:
    output = OUTPUT / f"{survey}_appendix.tex"
    output.parent.mkdir(exist_ok=True, parents=True)
    codebook = survey_data().codebook.filter(pl.col("survey") == survey)
    output.write_text(questions_appendix(codebook, QUESTION_SETS[survey]))
