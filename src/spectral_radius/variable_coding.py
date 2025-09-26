import polars as pl

# TODO: determine appropriate scaling for these


def scale_to_pm_1(x: pl.Expr) -> pl.Expr:
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1


def agreement(col: pl.Expr) -> pl.Expr:
    return col.replace_strict(
        {
            1: +1,  # strongly agree
            2: +0.5,  # TODO:  should these middle values be evenly spaced?
            3: -0.5,
            4: -1,  # strongly disagree
        }
    )


def current_level(col: pl.Expr) -> pl.Expr:
    return col.replace_strict(
        {
            1: +1,  # too little
            2: 0,
            3: -1,  # too much
        }
    )


def yes_or_no(col: pl.Expr) -> pl.Expr:
    return col.replace_strict(
        {
            1: +1,  # yes
            2: -1,  # no
        }
    )
