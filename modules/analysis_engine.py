
import re

import pandas as pd


def _normalize(text: str) -> str:
    """Normalize text for simple question matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _find_numeric_column(df: pd.DataFrame, question: str):
    """Find a numeric column mentioned in the user's question."""
    normalized_question = _normalize(question)

    aliases = {
        "revenue": ["revenue", "sales", "total sales"],
        "profit": ["profit", "earnings"],
        "units_sold": ["units sold", "quantity", "units", "volume"],
        "unit_price": ["unit price", "price"],
        "unit_cost": ["unit cost", "cost"],
    }

    for column in df.select_dtypes(include="number").columns:
        column_name = _normalize(column)
        possible_names = [column_name]
        possible_names.extend(aliases.get(column_name, []))

        for name in sorted(possible_names, key=len, reverse=True):
            normalized_name = _normalize(name)

            if normalized_name and re.search(
                rf"\b{re.escape(normalized_name)}\b",
                normalized_question,
            ):
                return column

    return None


def _find_group_column(df: pd.DataFrame, question: str):
    """Find a categorical column mentioned in the question."""
    normalized_question = _normalize(question)

    for column in df.columns:
        column_name = _normalize(column)

        if column_name and re.search(
            rf"\b{re.escape(column_name)}\b",
            normalized_question,
        ):
            if not pd.api.types.is_numeric_dtype(df[column]):
                return column

    return None


def get_dataset_summary(df: pd.DataFrame) -> str:
    """Return a concise summary of a dataset."""
    if df.empty:
        return "The dataset contains no rows to analyze."

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    summary = [
        f"The dataset contains {len(df)} rows and {len(df.columns)} columns.",
        f"Numeric columns: {', '.join(numeric_columns) or 'None'}.",
        f"Categorical columns: {', '.join(categorical_columns) or 'None'}.",
        f"Missing cells: {int(df.isna().sum().sum())}.",
        f"Duplicate rows: {int(df.duplicated().sum())}.",
    ]

    return "\n".join(summary)


def answer_question(df: pd.DataFrame, question: str) -> str:
    """Answer supported analytical questions using Pandas."""
    if df.empty:
        return "The dataset is empty, so I cannot calculate an answer."

    if not question.strip():
        return "Please enter a question about your dataset."

    normalized_question = _normalize(question)
    words = normalized_question.split()

    # Count records.
    if any(word in words for word in ["rows", "records", "entries"]):
        if any(
            word in words
            for word in ["how", "many", "count", "number"]
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
        ]
    ):
        return get_dataset_summary(df)

    numeric_column = _find_numeric_column(df, question)
    group_column = _find_group_column(df, question)

    grouped_question = (
        any(
            word in words
            for word in ["by", "per", "across", "among", "each"]
        )
        or any(
            phrase in normalized_question
            for phrase in [
                "which region",
                "what region",
                "which category",
                "what category",
                "which product",
                "what product",
                "which department",
                "what department",
            ]
        )
    )

    # Compare a numeric column across groups.
    if numeric_column and group_column and grouped_question:
        grouped = (
            df.groupby(group_column, dropna=False)[numeric_column]
            .sum()
            .sort_values(ascending=False)
        )

        if grouped.empty:
            return "There is no data available for that comparison."

        best_group = grouped.index[0]
        best_value = grouped.iloc[0]

        return (
            f"Total {numeric_column} by {group_column}:\n"
            f"{grouped.to_string()}\n\n"
            f"Highest {numeric_column}: {best_value:,.2f}, "
            f"for {group_column}: {best_group}."
        )

    # Questions about a specific numeric column.
    if numeric_column:
        values = pd.to_numeric(
            df[numeric_column], errors="coerce"
        ).dropna()

        if values.empty:
            return (
                f"Column '{numeric_column}' has no numeric values "
                "to analyze."
            )

        if any(
            word in words for word in ["average", "avg", "mean"]
        ):
            return f"Average {numeric_column}: {values.mean():,.2f}"

        if any(
            word in words
            for word in [
                "maximum", "max", "highest", "largest", "most"
            ]
        ):
            return f"Highest {numeric_column}: {values.max():,.2f}"

        if any(
            word in words
            for word in [
                "minimum", "min", "lowest", "smallest", "least"
            ]
        ):
            return f"Lowest {numeric_column}: {values.min():,.2f}"

        if any(word in words for word in ["total", "sum"]):
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
        "I couldn't identify a supported calculation from that question. "
        "Try asking for total revenue, average profit, highest units sold, "
        "or revenue by region."
    )
