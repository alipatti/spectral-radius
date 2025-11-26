from itertools import product
from pathlib import Path
import string


import polars as pl
import inflection
import gabriel

from spectral_radius.gss.data import get_all_gss_variables, load_gss_data

OPINION_CATEGORIES = {
    "Free Speech": [
        f"{a}{b}"
        for a, b in product(
            ("spk", "lib", "col"),
            ("ath", "mil", "rac", "homo", "mslm"),
        )
    ],
    "Government Spending": [
        "advfront",
    ]
    + [
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
    "Race": [
        "wrkwayup",
        "natrace",
        *(f"{prefix}rac" for prefix in ("spk", "col", "lib")),
        *(f"racdif{i}" for i in (1, 2, 3, 4)),
    ],
    "Welfare": [
        # "aidhouse", # not available enough
        "helpsick",
        "helppoor",
        "eqwlth",
        "natfare",
    ],
    "Police and the Judicial System": [
        *(f"pol{action}" for action in ("hitok", "abuse", "murdr", "escap", "attak")),
        "courts",
        "cappun",
        "natcrime",
        "conjudge",
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
    # "Miscellaneous": [
    #     # guns
    #     "gunlaw",
    #     # religion
    #     "prayer",
    #     "postlife",
    #     "god",
    #     # marijuana
    #     "grass",
    #     # science
    #     # immigration
    #     "letin1a",
    #     # american dream
    #     "goodlife",
    # ],
}


OPINION_VARIABLES = [v for vs in OPINION_CATEGORIES.values() for v in vs]


async def categorize_variables():
    years_available = pl.col("years").list.eval(
        pl.element()
        .struct.field("year")
        .filter(
            pl.element().struct.field("isQuestionAvailable").ne("unavailable"),
            pl.element().struct.field("year") >= 1990,
        )
        .mod(100)
    )

    variables = get_all_gss_variables()

    good = variables.filter(years_available.list.len() > 10).with_columns(
        years=years_available,
    )

    for_llm = pl.format(
        "Name: {}\nShort description: {}\nQuestion text: {}",
        "name",
        "description",
        "survey_question",
    ).alias("for_llm")
    for_gabriel = good.select("name", for_llm)

    instructions = """
    These are questions from the General Social Survey.
    Do not classify demographic or logistical questions about things like year or location of survey or the respondent's occupation, age, year, workforce status, family members, etc.
    Do not classify questions that ask about a static characteristic like the respondent's religion or whether the respondent partakes in a certain activity.
    """

    category_descriptions = {
        "Opinion Question": "A question asking the respondent's opinion on some topic.",
        "Role of Government": "A question asking the respondent's opinion about taxation, welfare, wealth redistribution, income inequality, government responsibility for improving living standards and providing public services, whether the government spends too much or too little on a particular service, or a similar topic.",
        "Race and Affirmative Action": "A question asking the respondent's opinion on race relations, discrimination, government aid to Black Americans, race-based affirmative action, racial preference in hiring or promotion policies, or another race-related opinion.",
        "Gender Roles and Feminism": "A question asking the respondent's opinion concerning women’s roles in politics, family, and work, or attitudes toward gender equality and affirmative action for women.",
        "Sex, Birth Control, and Abortion": "A question asking the respondent's opinion about sexual morality, contraception, abortion rights, homosexuality, pornography, when it is okay to have sex, sex education, or another sexual topic.",
        "Religion and Moral Traditionalism": "A question asking the respondent's opinion on religious beliefs, practice, interpretation of scripture, moral traditionalism, and religion in public life.",
        "Free Speech and Civil Liberties": "A question asking the respondent's opinion on either of (1) personal autonomy, e.g. assisted suicide OR (2) tolerance for controversial or unpopular speech, teaching, or publications.",
        "Law, Order, and Criminal Justice": "A question asking the respondent's opinion on policing, laws, crime control, punishment, the death penalty, drug laws, or another related topic.",
        "Trust in Institutions": "A question asking the respondent's opinion on confidence in major institutions, like the media, government, military, or another institution.",
    }

    classification_results = await gabriel.classify(
        for_gabriel.to_pandas(),
        "for_llm",
        labels=category_descriptions,
        save_dir=".gabriel/gss_classification",
        additional_instructions=instructions,
        # reset_files=True,
    )

    labeled_questions = good.join(
        pl.from_pandas(classification_results).select("name", "predicted_classes"),
        how="inner",
        on="name",
        validate="1:1",
    ).select("name", "description", "survey_question", "predicted_classes")

    list(
        labeled_questions.sort(
            pl.col("predicted_classes").list.len().mul(-1)
        ).iter_rows()
    )


def escape_tex(s: str) -> str:
    return s.replace("&", r"\&").replace("\n", r"\\").replace(". . .", r"$\ldots$ ")


def format_question(name, desc, text, responses) -> str:
    return rf"""
    \textsc{{{name}}} \hfill \textit{{({escape_tex(inflection.humanize(desc))})}}
    \begin{{quote}}
        \footnotesize
        {escape_tex(text)} 

        \textit{{Possible responses}}: {escape_tex(responses)}
    \end{{quote}}
    """


def get_response_labels():
    metadata = load_gss_data(use_cache=True)[1]

    return {
        name: ", ".join(
            f"``{v}'' ({k})" for k, v in labels.items() if isinstance(k, int)
        )
        for name, labels in metadata.variable_value_labels.items()
    }


def make_questions_appendix(
    output=Path("./paper/generated/gss_appendix.tex"),
):
    survey_questions = get_all_gss_variables().select(
        "name",
        "description",
        "survey_question",
    )

    responses = get_response_labels()

    df = (
        pl.DataFrame(
            {"cat": cat, "name": v}
            for cat, vs in OPINION_CATEGORIES.items()
            for v in vs
        )
        .join(survey_questions, "name", validate="m:1")
        .with_columns(
            pl.col("name").replace_strict(responses, default=None).alias("responses")
        )
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
