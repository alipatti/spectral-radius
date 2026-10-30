"""
Exploratory llm classification of GSS questions into topics.

Not part of the replication: the paper's question sets
(`surveys.QUESTION_SETS`) were chosen by hand.
"""

import gabriel
import polars as pl

from spectral_radius.surveys import GSS
from spectral_radius.surveys.gss import QUESTIONS_FILE

INSTRUCTIONS = """
These are questions from the General Social Survey.
Do not classify demographic or logistical questions about things like year or location of survey or the respondent's occupation, age, year, workforce status, family members, etc.
Do not classify questions that ask about a static characteristic like the respondent's religion or whether the respondent partakes in a certain activity.
"""

CATEGORY_DESCRIPTIONS = {
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


async def categorize_variables() -> pl.DataFrame:
    """Classify the questions asked in more than ten years since 1990."""
    years = pl.element().struct.field("year")
    available = pl.element().struct.field("isQuestionAvailable") != "unavailable"
    years_available = pl.col("years").list.eval(
        years.filter(available, years >= 1990).mod(100)
    )

    variables = pl.read_parquet(GSS().fetch() / QUESTIONS_FILE).with_columns(
        pl.col("surveyQuestion").list.first().alias("survey_question")
    )
    good = variables.filter(years_available.list.len() > 10).with_columns(
        years=years_available,
    )

    for_llm = pl.format(
        "Name: {}\nShort description: {}\nQuestion text: {}",
        "name",
        "description",
        "survey_question",
    ).alias("for_llm")

    classification_results = await gabriel.classify(
        good.select("name", for_llm).to_pandas(),
        "for_llm",
        labels=CATEGORY_DESCRIPTIONS,
        save_dir=".gabriel/gss_classification",
        additional_instructions=INSTRUCTIONS,
    )

    return (
        good.join(
            pl.from_pandas(classification_results).select("name", "predicted_classes"),
            how="inner",
            on="name",
            validate="1:1",
        )
        .select("name", "description", "survey_question", "predicted_classes")
        .sort(pl.col("predicted_classes").list.len(), descending=True)
    )
