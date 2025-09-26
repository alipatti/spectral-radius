from itertools import product

VARIABLE_CATEGORIES = {
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


ALL_VARIABLES = [v for vs in VARIABLE_CATEGORIES.values() for v in vs]
