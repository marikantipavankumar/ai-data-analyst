
import re
import pandas as pd

from modules.analysis_engine import (
    _find_group_column,
    _find_numeric_column,
    _normalize,
)


def resolve_follow_up(
    df: pd.DataFrame,
    question: str,
    previous_question: str | None,
) -> tuple[pd.DataFrame, str, str | None]:
    """
    Resolve a follow-up question using the previous question.

    Example:
    Previous: What is the average revenue by region?
    Follow-up: What about East region?

    Returns the filtered dataframe, the question to analyze,
    and a note describing the context used.
    """

    if not question or not question.strip():
        return df, question, None

    if not previous_question or not previous_question.strip():
        return df, question, None

    # Identify the grouping column from the previous question.
    group_column = _find_group_column(df, previous_question)

    if group_column is None:
        return df, question, None

    normalized_question = _normalize(question)

    # Find a dataset category mentioned in the follow-up.
    category_values = (
        df[group_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    matched_value = None

    # Check longer category names first.
    for value in sorted(category_values, key=len, reverse=True):
        normalized_value = _normalize(value)

        if normalized_value and re.search(
            rf"\b{re.escape(normalized_value)}\b",
            normalized_question,
        ):
            matched_value = value
            break

    if matched_value is None:
        return df, question, None

    # Filter the dataset to the requested category.
    filtered_df = df[
        df[group_column].astype(str).str.casefold()
        == matched_value.casefold()
    ]

    if filtered_df.empty:
        return df, question, None

    # If the follow-up specifies a new metric or operation, use it.
    # Otherwise, reuse the previous question.
    current_metric = _find_numeric_column(df, question)
    normalized_words = set(normalized_question.split())

    operation_words = {
        "average", "avg", "mean",
        "total", "sum",
        "highest", "maximum", "max", "largest", "most",
        "lowest", "minimum", "min", "smallest", "least",
    }

    has_operation = bool(normalized_words.intersection(operation_words))

    if current_metric or has_operation:
        question_to_analyze = question
    else:
        question_to_analyze = previous_question

    context_note = (
        f"Using the previous analysis, filtered "
        f"'{group_column}' to '{matched_value}'."
    )

    return filtered_df, question_to_analyze, context_note
