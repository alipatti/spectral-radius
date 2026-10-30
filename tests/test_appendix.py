from helpers import frame

from spectral_radius.appendix import questions_appendix
from spectral_radius.surveys import Codebook


def codebook():
    return frame(
        Codebook,
        [
            {
                "survey": "gss",
                "variable": "q",
                "description": "abortion if wanted",
                "question": "Should abortion be legal? . . . & why",
                "years": [2000],
                "labels": {"value": [1, 2], "label": ["yes", "no"]},
            },
            {
                "survey": "gss",
                "variable": "scale",
                "description": "a scale",
                "question": "Where are you?",
                "years": [2000],
                "labels": {"value": [], "label": []},
            },
        ],
    )


def test_appendix_lists_question_text_and_labels():
    """Show each question's escaped text and its labeled responses."""
    tex = questions_appendix(codebook(), {"Topic": ["q"]})

    assert r"\subsection{Topic}" in tex
    assert r"\textsc{q} \hfill \textit{(Abortion if wanted)}" in tex
    assert r"Should abortion be legal? $\ldots$  \& why" in tex
    assert "``yes'' (1), ``no'' (2)" in tex


def test_appendix_sorts_categories_and_variables():
    """Order categories, and variables within them, by name."""
    tex = questions_appendix(codebook(), {"B": ["scale", "q"], "A": ["q"]})

    assert tex.index(r"\subsection{A}") < tex.index(r"\subsection{B}")
    in_b = tex[tex.index(r"\subsection{B}") :]
    assert in_b.index(r"\textsc{q}") < in_b.index(r"\textsc{scale}")


def test_appendix_leaves_unlabeled_responses_empty():
    """List no responses for a variable without value labels."""
    tex = questions_appendix(codebook(), {"Topic": ["scale"]})

    assert r"\textit{Possible responses}: " + "\n" in tex
