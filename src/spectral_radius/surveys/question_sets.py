"""Question sets: named categories of opinion variables, per survey."""

from itertools import product

GSS_OPINION_CATEGORIES: dict[str, list[str]] = {
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


QUESTION_SETS: dict[str, dict[str, list[str]]] = {"gss": GSS_OPINION_CATEGORIES}


# the variables of each survey's question sets with categorical scales
# that the analysis nonetheless encodes as if ordinal (see `wide`)
ALLOWED_CATEGORICAL: dict[str, frozenset[str]] = {
    # NOTE: the codes of `courts` (1 too harshly, 2 not harshly enough,
    # 3 about right) aren't in the order of its answers; it is encoded by code,
    # as the paper always has, pending a decision to recode or drop it
    "gss": frozenset({"courts"}),
}


def question_variables(survey: str) -> list[str]:
    """Unique variables named by any category of a survey, in first-seen order."""
    return list(dict.fromkeys(v for vs in QUESTION_SETS[survey].values() for v in vs))
