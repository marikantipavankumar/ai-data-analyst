
import re

import pandas as pd


def _normalize(text: str) -> str:
    """Normalize text for consistent question matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _find_numeric_column(df: pd.DataFrame, question: str):
    """Find a numeric column mentioned in the question."""
    normalized_question = _normalize(question)

    for column in df.select_dtypes(include="number").columns:
        column_name = _normalize(column)

        if column_name and re.search(
            rf"\b{re.escape(column_name)}\b",
            normalized_question,
        ):
            return column

    return None


def _find_group_column(df: pd.DataFrame, question: str):
    """Find a non-numeric column mentioned in the question."""
    normalized_question = _normalize(question)

    # Prefer longer column names to avoid partial matches.
    columns = sorted(df.columns, key=lambda col: len(_normalize(col)), reverse=True)

    for column in columns:
        column_name = _normalize(column)

        if column_name and re.search(
            rf"\b{re.escape(column_name)}\b",
            normalized_question,
        ):
            if not pd.api.types.is_numeric_dtype(df[column]):
                return column

    return None


def get_dataset_summary(df: pd.DataFrame) -> str:
    """Return a concise summary of the uploaded dataset."""
    if df.empty:
        return "The dataset contains no rows to analyze."

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return "\n".join([
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Numeric columns: {', '.join(map(str, numeric_columns)) or 'None'}",
        f"Categorical columns: {', '.join(map(str, categorical_columns)) or 'None'}",
        f"Missing cells: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
    ])


def answer_question(df: pd.DataFrame, question: str) -> str:
    """Answer supported analytical questions using Pandas."""
    if not question or not question.strip():
        return "Please enter a question about your dataset."

    if df.empty:
        return "The dataset is empty, so I cannot calculate an answer."

    normalized_question = _normalize(question)
    words = normalized_question.split()
    word_set = set(words)

    # Count records.
    if (
        word_set.intersection({"rows", "records", "entries"})
        and word_set.intersection({"how", "many", "count", "number"})
    ):
        return f"The dataset contains {len(df)} rows."

    # Dataset overview.
    if any(
        phrase in normalized_question
        for phrase in [
            "summarize",
            "summary",
            "overview",
            "describe the dataset",
            "describe the data",
        ]
    ):
        return get_dataset_summary(df)

    # Identify the requested calculation.
    if word_set.intersection({"average", "avg", "mean"}):
        operation = "mean"
    elif word_set.intersection({"highest", "maximum", "max", "largest", "most"}):
        operation = "max"
    elif word_set.intersection({"lowest", "minimum", "min", "smallest", "least"}):
        operation = "min"
    elif word_set.intersection({"total", "sum"}):
        operation = "sum"
    else:
        operation = "summary"

    numeric_column = _find_numeric_column(df, question)
    group_column = _find_group_column(df, question)

    # Detect grouped questions, including "which student has highest marks".
    grouped_question = (
        word_set.intersection({"by", "per", "across", "among", "each"})
        or (
            group_column is not None
            and word_set.intersection({"which", "what"})
            and operation in {"max", "min"}
        )
    )

    if numeric_column and group_column and grouped_question:
        values = pd.to_numeric(df[numeric_column], errors="coerce")

        grouped_data = values.groupby(
            df[group_column],
            dropna=False,
        )

        if operation == "mean":
            grouped = grouped_data.mean()
            label = "Average"
        elif operation == "max":
            grouped = grouped_data.max()
            label = "Maximum"
        elif operation == "min":
            grouped = grouped_data.min()
            label = "Minimum"
        else:
            grouped = grouped_data.sum(min_count=1)
            label = "Total"

        grouped = grouped.dropna()

        if grouped.empty:
            return "There is no valid data available for that comparison."

        # For a question asking which group is highest/lowest,
        # return the winning group rather than only the numeric value.
        if operation in {"max", "min"} and word_set.intersection(
            {"which", "what", "highest", "lowest", "maximum", "minimum"}
        ):
            target_value = grouped.max() if operation == "max" else grouped.min()
            target_groups = grouped[grouped == target_value]

            group_name = target_groups.index[0]
            return (
                f"{group_column} with the {label.lower()} "
                f"{numeric_column}: {group_name} "
                f"({target_value:,.2f})."
            )

        # Show every group's calculation.
        grouped = grouped.sort_values(ascending=False)

        result = grouped.to_string()

        return (
            f"{label} {numeric_column} by {group_column}:\n"
            f"{result}"
        )

    # Answer questions about a single numeric column.
    if numeric_column:
        values = pd.to_numeric(
            df[numeric_column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return (
                f"Column '{numeric_column}' has no valid numeric values."
            )

        if operation == "mean":
            return f"Average {numeric_column}: {values.mean():,.2f}"

        if operation == "max":
            return f"Highest {numeric_column}: {values.max():,.2f}"

        if operation == "min":
            return f"Lowest {numeric_column}: {values.min():,.2f}"

        if operation == "sum":
            return f"Total {numeric_column}: {values.sum():,.2f}"

        return (
            f"Summary for {numeric_column}:\n"
            f"Count: {values.count()}\n"
            f"Total: {values.sum():,.2f}\n"
            f"Average: {values.mean():,.2f}\n"
            f"Minimum: {values.min():,.2f}\n"
            f"Maximum: {values.max():,.2f}"
        )

    return (
        "I couldn't identify the columns or calculation in your question. "
        "Mention a column name from your dataset and try a question such as "
        "'What is the average [column]?' or '[column] by [category column]'."
    )
