"""
Response scales, declared once and shared by every survey.

A `Scale` lists a question's answers as levels in their substantive order,
from one end of the scale to the other, whatever the order of their raw codes.
Each level is a raw code and the label variants a source may publish for it;
a level without labels (e.g. the unlabeled middle of a 1-to-7 scale)
is declared with `unlabeled`.
A scale may also declare `off_scale` codes:
labeled answers that aren't on the scale (e.g. a volunteered "depends"),
which sources read as missing, of kind `off_scale`.
Each source assigns a scale to every variable it reads;
`require_labels_match` checks the assignment against the published labels.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import polars as pl

type ScaleKind = Literal["binary", "ordinal"]


def normalize_label(label: str) -> str:
    """A label compared case-insensitively, with its whitespace collapsed."""
    return " ".join(label.split()).casefold()


@dataclass(frozen=True)
class Level:
    """A raw code and the labels a source may publish for it (none if unlabeled)."""

    code: int
    labels: tuple[str, ...]

    def accepts(self, label: str) -> bool:
        return normalize_label(label) in {normalize_label(v) for v in self.labels}


def level(code: int, *labels: str) -> Level:
    """A labeled level, accepting any of `labels`."""
    if not labels:
        raise ValueError(f"Level {code} has no labels; declare it with `unlabeled`")

    return Level(code, labels)


def unlabeled(code: int) -> Level:
    """A level the source publishes no label for."""
    return Level(code, ())


@dataclass(frozen=True)
class Scale:
    """
    A named scale: its `levels` in substantive order, and its known `off_scale` codes.

    Codes are unique across levels and off-scale codes,
    there are at least two levels,
    and every off-scale code is labeled.
    """

    name: str
    levels: tuple[Level, ...]
    off_scale: tuple[Level, ...] = ()

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError(f"Scale {self.name!r} has fewer than two levels")

        counts = Counter(lvl.code for lvl in (*self.levels, *self.off_scale))

        if repeated := sorted(code for code, n in counts.items() if n > 1):
            raise ValueError(f"Scale {self.name!r} repeats codes {repeated}")

        if unlabeled_off := [lvl.code for lvl in self.off_scale if not lvl.labels]:
            raise ValueError(
                f"Scale {self.name!r} has unlabeled off-scale codes {unlabeled_off}"
            )

    @property
    def codes(self) -> list[int]:
        """The codes of the levels, in scale order."""
        return [lvl.code for lvl in self.levels]

    @property
    def off_scale_codes(self) -> list[int]:
        return [lvl.code for lvl in self.off_scale]

    @property
    def kind(self) -> ScaleKind:
        return "binary" if len(self.levels) == 2 else "ordinal"


def _registry(scales: Iterable[Scale]) -> Mapping[str, Scale]:
    scales = list(scales)
    names = Counter(s.name for s in scales)

    if repeated := sorted(name for name, n in names.items() if n > 1):
        raise ValueError(f"Scales share names {repeated}")

    return MappingProxyType({s.name: s for s in scales})


YES_NO = Scale("yes_no", (level(1, "yes"), level(2, "no")))
FAVOR_OPPOSE = Scale(
    "favor_oppose",
    (level(1, "favor"), level(2, "oppose")),
    # volunteered in the gss's `sexeduc`, in 1974 only
    off_scale=(level(3, "depends on age/grade (vol.)"),),
)
ALLOWED_TO_SPEAK = Scale(
    "allowed_to_speak",
    (level(1, "yes, allowed to speak"), level(2, "not allowed")),
)
ALLOWED_TO_TEACH = Scale(
    "allowed_to_teach",
    (level(4, "yes, allowed to teach"), level(5, "not allowed")),
)
REMOVE_BOOK = Scale("remove_book", (level(1, "remove"), level(2, "not remove")))
TOO_LITTLE_TO_TOO_MUCH = Scale(
    "too_little_to_too_much",
    (level(1, "too little"), level(2, "about right"), level(3, "too much")),
)
AGREE_4 = Scale(
    "agree_4",
    (
        level(1, "strongly agree"),
        level(2, "agree"),
        level(3, "disagree"),
        level(4, "strongly disagree"),
    ),
)
AGREE_5 = Scale(
    "agree_5",
    (
        level(1, "strongly agree", "agree strongly"),
        level(2, "agree", "agree somewhat"),
        level(3, "neither agree nor disagree"),
        level(4, "disagree", "disagree somewhat"),
        level(5, "strongly disagree", "disagree strongly"),
    ),
)
FAVOR_OPPOSE_4 = Scale(
    "favor_oppose_4",
    (
        level(1, "strongly favor", "strongly favors"),
        level(2, "not strongly favor", "not strongly favors"),
        level(3, "not strongly oppose", "not strongly opposes"),
        level(4, "strongly oppose", "strongly opposes"),
    ),
)
ALWAYS_TO_NOT_WRONG = Scale(
    "always_to_not_wrong",
    (
        level(1, "always wrong"),
        level(2, "almost always wrong"),
        level(3, "wrong only sometimes"),
        level(4, "not wrong at all"),
    ),
    # given only in 1973 and 1974
    off_scale=(level(5, "other"),),
)
GOVERNMENT_TO_PEOPLE = Scale(
    "government_to_people",
    (
        level(1, "government should help"),
        unlabeled(2),
        level(3, "agree with both"),
        unlabeled(4),
        level(5, "people should care for themselves", "no special treatment"),
    ),
)
REDUCE_INCOME_DIFFERENCES = Scale(
    "reduce_income_differences",
    (
        level(1, "the government should reduce income differences"),
        *map(unlabeled, range(2, 7)),
        level(
            7,
            "the government should not concern itself with reducing income differences",
        ),
    ),
)
CONFIDENCE = Scale(
    "confidence",
    (level(1, "a great deal"), level(2, "only some"), level(3, "hardly any")),
)
# the codes aren't in the order of the answers: "about right" lies between
COURTS_HARSHNESS = Scale(
    "courts_harshness",
    (
        level(1, "too harshly"),
        level(3, "about right"),
        level(2, "not harshly enough"),
    ),
)

SCALES: Mapping[str, Scale] = _registry(
    [
        YES_NO,
        FAVOR_OPPOSE,
        ALLOWED_TO_SPEAK,
        ALLOWED_TO_TEACH,
        REMOVE_BOOK,
        TOO_LITTLE_TO_TOO_MUCH,
        AGREE_4,
        AGREE_5,
        FAVOR_OPPOSE_4,
        ALWAYS_TO_NOT_WRONG,
        GOVERNMENT_TO_PEOPLE,
        REDUCE_INCOME_DIFFERENCES,
        CONFIDENCE,
        COURTS_HARSHNESS,
    ]
)


def label_mismatches(
    scale: Scale,
    values: Sequence[int],
    labels: Sequence[str],
) -> list[str]:
    """
    How a variable's published value labels disagree with its `scale`, if at all.

    Each published label must be one of the variants its code accepts,
    whether the code is a level or an off-scale code,
    every published code must be one of those,
    and every labeled level must be published.
    """
    declared = {lvl.code: lvl for lvl in (*scale.levels, *scale.off_scale)}
    published = dict(zip(values, labels, strict=True))

    problems = [
        f"code {code} is labeled {label!r}, which isn't on scale {scale.name!r}"
        if code not in declared
        else f"code {code} is labeled {label!r}, "
        f"but scale {scale.name!r} accepts {list(declared[code].labels)}"
        for code, label in published.items()
        if code not in declared or not declared[code].accepts(label)
    ]
    problems += [
        f"code {lvl.code} is unlabeled, "
        f"but scale {scale.name!r} labels it {list(lvl.labels)}"
        for lvl in scale.levels
        if lvl.labels and lvl.code not in published
    ]

    return problems


def require_labels_match(codebook: pl.DataFrame, scales: Mapping[str, Scale]) -> None:
    """
    Require the codebook's variables read to be those of `scales`, labeled as such.

    `codebook` is one survey's codebook and `scales` maps each variable it reads
    to its scale; raises naming every variable, code, and label that disagree.
    """
    read = codebook.filter("in_responses").select(
        pl.col("variable").cast(pl.String),
        pl.col("labels").struct.field("value"),
        pl.col("labels").struct.field("label"),
    )
    read_variables = set(read["variable"])

    if read_variables != scales.keys():
        raise ValueError(
            "The codebook's variables read differ from those with scales: "
            f"read only {sorted(read_variables - scales.keys())}, "
            f"scaled only {sorted(scales.keys() - read_variables)}"
        )

    problems = [
        f"{variable}: {problem}"
        for variable, values, labels in read.iter_rows()
        for problem in label_mismatches(scales[variable], values, labels)
    ]

    if problems:
        raise ValueError(
            "Published value labels disagree with declared scales:\n"
            + "\n".join(problems)
        )
