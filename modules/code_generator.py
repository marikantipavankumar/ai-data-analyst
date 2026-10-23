
import re
import pandas as pd

from modules.analysis_engine import (
    _normalize,
    _find_numeric_column,
    _find_group_column,
)


def _quote_sql_identifier(name: str) -> str:
    """Quote an SQL identifier safely."""
    return '"' + str(name).replace('"', '""') + '"'


def _make_alias(operation: str, column: str) -> str:
    """Create a readable output alias for generated code."""
    safe_column = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        str(column),
    ).strip("_").lower()

    return f"{operation}_{safe_column}"


def generate_analysis_code(
    df: pd.DataFrame,
    question: str,
    table_name: str = "uploaded_data",
) -> dict[str, str]:
    """
    Generate Pandas and SQL examples for supported data questions.

    The returned code is for display and inspection; it is not executed.
    """

    if not question or not question.strip():
        return {
            "supported": "false",
            "message": "Please enter a question.",
            "pandas_code": "",
            "sql_code": "",
        }

    if df.empty:
        return {
            "supported": "false",
            "message": "The dataset is empty.",
            "pandas_code": "",
            "sql_code": "",
        }

    normalized = _normalize(question)
    words = set(normalized.split())

    # Handle questions asking for the number of rows.
    if (
        words.intersection({"rows", "records", "entries"})
        and words.intersection({"how", "many", "count", "number"})
    ):
        return {
            "supported": "true",
            "message": "Count the rows in the dataset.",
            "pandas_code": "result = len(df)",
            "sql_code": (
                f"SELECT COUNT(*) AS row_count\n"
                f"FROM {_quote_sql_identifier(table_name)};"
            ),
        }

    # Identify the requested aggregation.
    if words.intersection({"average", "avg", "mean"}):
        operation = "mean"
        operation_label = "average"
        sql_function = "AVG"
    elif words.intersection({"total", "sum"}):
        operation = "sum"
        operation_label = "total"
        sql_function = "SUM"
    elif words.intersection(
        {"highest", "maximum", "max", "largest", "most"}
    ):
        operation = "max"
        operation_label = "maximum"
        sql_function = "MAX"
    elif words.intersection(
        {"lowest", "minimum", "min", "smallest", "least"}
    ):
        operation = "min"
        operation_label = "minimum"
        sql_function = "MIN"
    else:
        return {
            "supported": "false",
            "message": (
                "I couldn't identify a supported calculation. "
                "Try asking for an average, total, maximum, minimum, "
                "or row count."
            ),
            "pandas_code": "",
            "sql_code": "",
        }

    # Identify the numeric metric in the question.
    numeric_column = _find_numeric_column(df, question)

    if numeric_column is None:
        return {
            "supported": "false",
            "message": (
                "Please mention a numeric column from your dataset, "
                "such as revenue or profit."
            ),
            "pandas_code": "",
            "sql_code": "",
        }

    # Identify a grouping column when the question requests one.
    group_column = _find_group_column(df, question)
    grouping_words = {"by", "per", "across", "among", "each"}

    is_grouped = (
        group_column is not None
        and bool(words.intersection(grouping_words))
    )

    pandas_column = repr(str(numeric_column))
    sql_column = _quote_sql_identifier(numeric_column)
    sql_table = _quote_sql_identifier(table_name)

    if is_grouped:
        pandas_group = repr(str(group_column))
        sql_group = _quote_sql_identifier(group_column)

        result_alias = _make_alias(
            operation_label,
            numeric_column,
        )
        sql_alias = _quote_sql_identifier(result_alias)

        ascending = operation == "min"
        sort_order = "ASC" if ascending else "DESC"

        pandas_code = (
            "result = (\n"
            f"    df.groupby({pandas_group}, dropna=False)"
            f"[{pandas_column}]\n"
            f"    .{operation}()\n"
            f"    .reset_index(name={result_alias!r})\n"
            f"    .sort_values({result_alias!r}, "
            f"ascending={ascending})\n"
            ")"
        )

        sql_code = (
            f"SELECT {sql_group},\n"
            f"       {sql_function}({sql_column}) AS {sql_alias}\n"
            f"FROM {sql_table}\n"
            f"GROUP BY {sql_group}\n"
            f"ORDER BY {sql_alias} {sort_order};"
        )

        message = (
            f"Calculate the {operation_label} of "
            f"'{numeric_column}' for each value of '{group_column}'."
        )

    else:
        pandas_code = (
            f"result = df[{pandas_column}].{operation}()"
        )

        result_alias = _make_alias(
            operation_label,
            numeric_column,
        )
        sql_alias = _quote_sql_identifier(result_alias)

        sql_code = (
            f"SELECT {sql_function}({sql_column}) AS {sql_alias}\n"
            f"FROM {sql_table};"
        )

        message = (
            f"Calculate the {operation_label} of "
            f"'{numeric_column}'."
        )

    return {
        "supported": "true",
        "message": message,
        "pandas_code": pandas_code,
        "sql_code": sql_code,
    }
