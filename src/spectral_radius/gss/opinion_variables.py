from itertools import product
from pathlib import Path

import polars as pl
import inflection
from great_tables import GT

from spectral_radius.gss.data import get_all_gss_variables

OPINION_CATEGORIES = {
    "Free Speech": [
        f"{a}{b}"
        for a, b in product(
            ("spk", "lib", "col"),
            ("ath", "mil", "rac", "homo", "mslm"),
        )
    ],
    "Government Spending": [
        f"nat{expense}"
        for expense in (
            "spac",
            "envir",
            "heal",
            "city",
            "crime",
            "drug",
            "educ",
            "race",
            "arms",
            "aid",
            "fare",
            "road",
            "soc",
            "mass",
            "park",
            # "spacy",
            # "enviy",
            # "healy",
            # "cityy",
            # "crimy",
            # "drugy",
            # "educy",
            # "racey",
            # "armsy",
            # "aidy",
            # "farey",
            # "sci",
            # "chld",
        )
    ],
    "Affirmative Action": [
        "helpblk",
        "affrmact",
        "fejobaff",
        "fehire",
    ],
    "Race": [f"racdif{i}" for i in (1, 2, 3, 4)]
    + [
        "wrkwayup",
    ],
    "Welfare": [
        # "aidhouse", # not available enough
        "helpsick",
        "helppoor",
        "eqwlth",
    ],
    "Police": [
        f"pol{action}" for action in ("hitok", "abuse", "murdr", "escap", "attak")
    ],
    "Sex and Birth Control": [
        # sex
        "premarsx",
        "xmarsex",
        "homosex",
        # "pornlaw",
        "sexeduc",
        "teensex",
        # birth control
        "pillok",
    ],
    "Abortion": [
        # abortion
        f"ab{reason}"
        for reason in ("defect", "nomore", "hlth", "poor", "rape", "single", "any")
    ],
    "Miscellaneous": [
        # political affiliation
        # "partyid",
        # "polviews",
        # guns
        "gunlaw",
        # religion
        "prayer",
        "postlife",
        "god",
        # marijuana
        "grass",
        # judicial/penal system
        "courts",
        "cappun",
        # science
        "advfront",
        # immigration
        "letin1a",
        # american dream
        "goodlife",
    ],
}


OPINION_VARIABLES = [v for vs in OPINION_CATEGORIES.values() for v in vs]


def escape_tex(s: str) -> str:
    return s.replace("&", r"\&").replace("\n", r"\\").replace(". . .", r"$\ldots$ ")


def format_question(name, desc, text) -> str:
    return rf"\textsc{{{name}}} \hfill \textit{{({escape_tex(inflection.humanize(desc))})}} \begin{{quote}} \footnotesize {escape_tex(text)} \end{{quote}}"


def make_questions_appendix(
    output=Path("./paper/generated/gss_appendix.tex"),
):
    survey_questions = get_all_gss_variables().select(
        "name",
        "description",
        "survey_question",
    )

    df = (
        pl.DataFrame(
            {"cat": cat, "name": v}
            for cat, vs in OPINION_CATEGORIES.items()
            for v in vs
        )
        .join(survey_questions, "name", validate="1:1")
        .sort("cat", "name")
    )

    categories = df.partition_by(
        "cat", maintain_order=True, as_dict=True, include_key=False
    )

    content = "\n".join(
        rf"\subsection{{{category}}}"
        + "\n\n".join(format_question(*q) for q in c.iter_rows())
        for (category,), c in categories.items()
    )

    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(content)

if __name__ == "__main__":
    make_questions_appendix()
