from modules.analysis_engine import (
    _normalize,
    _find_numeric_column,
    _find_group_column,
)


def explain_analysis(df, question: str) -> str:
    """Explain the calculation inferred from a data question."""

    if not question or not question.strip():
        return "No question was provided."

    normalized = _normalize(question)
    words = set(normalized.split())

    numeric_column = _find_numeric_column(df, question)
    group_column = _find_group_column(df, question)

    if words.intersection({"average", "avg", "mean"}):
        operation = "mean"
        description = "Calculate the arithmetic mean (average)."
    elif words.intersection({"total", "sum"}):
        operation = "sum"
        description = "Add the valid numeric values."
    elif words.intersection({"highest", "maximum", "max", "largest", "most"}):
        operation = "max"
        description = "Find the maximum value."
    elif words.intersection({"lowest", "minimum", "min", "smallest", "least"}):
        operation = "min"
        description = "Find the minimum value."
    else:
        operation = "summary"
        description = "Calculate a descriptive summary."

    lines = [
        f"Question: {question}",
        f"Calculation method: {description}",
    ]

    if numeric_column:
        lines.append(f"Metric column: {numeric_column}")
    else:
        lines.append("Metric column: No numeric column identified.")

    is_grouped = (
        group_column is not None
        and bool(words.intersection({"by", "per", "across", "among", "each"}))
    )

    if is_grouped:
        lines.append(f"Grouping column: {group_column}")
        lines.append(
            f"Calculation: {operation} of '{numeric_column}' "
            f"for each value in '{group_column}'."
        )

        if df[group_column].isna().any():
            lines.append(
                f"Missing '{group_column}' values are retained "
                "as a separate group."
            )
    else:
        lines.append("Grouping: No grouping identified.")

    if numeric_column:
        missing_count = int(
            df[numeric_column].isna().sum()
        )
        lines.append(
            f"Missing values in '{numeric_column}': {missing_count}"
        )
        lines.append(
            "Missing numeric values are excluded from this calculation."
        )

    return "\n".join(lines)
