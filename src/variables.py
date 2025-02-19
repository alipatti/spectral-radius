from itertools import product

subset_to_study = dict(
    freedom_of_speech=[
        f"{a}{b}"
        for a, b in product(
            ("spk", "lib", "col"),
            ("ath", "mil", "rac", "homo", "mslm"),
        )
    ],
    reproductive_rights=[
        "pillok",
    ]
    + [
        f"ab{reason}"
        for reason in ("defect", "nomore", "hlth", "poor", "rape", "single", "any")
    ],
    government_spending=[
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
            "sci",
            "chld",
        )
    ],
    affirmative_action=[
        "helpblk",
        "affrmact",
        "fejobaff",
        "fehire",
    ],
    race=[f"racdif{i}" for i in (1, 2, 3, 4)]
    + [
        "wrkwayup",
    ],
    welfare=[
        "aidhouse",
        "helpsick",
        "helppoor",
        "eqwlth",
    ],
    police=[f"pol{action}" for action in ("hitok", "abuse", "murdr", "scap", "attaak")],
    sex=[
        "premarsx",
        "xmarsex",
        "homosex",
        "pornlaw",
        "sexeduc",
        "teensex",
    ],
    misc=[
        # political affiliation
        "partyid",
        "polviews",
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
        "advfront"
        # immigration
        "letin1a",
        # american dream
        "goodlife",
    ],
)
