
import pandas as pd


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect unusually low or high values in numeric columns
    using the Interquartile Range (IQR) method.
    """
    anomalies = []

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        # IQR is not useful when fewer than four observations exist.
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        unusual_values = values[
            (values < lower_bound) | (values > upper_bound)
        ]

        for row_index, value in unusual_values.items():
            direction = (
                "unusually low"
                if value < lower_bound
                else "unusually high"
            )

            anomalies.append({
                "Row Number": row_index + 1,
                "Column": column,
                "Value": value,
                "Lower Bound": round(lower_bound, 2),
                "Upper Bound": round(upper_bound, 2),
                "Explanation": (
                    f"{value:,.2f} is {direction}. "
                    f"The expected IQR range is "
                    f"{lower_bound:,.2f} to {upper_bound:,.2f}."
                ),
            })

    result_columns = [
        "Row Number",
        "Column",
        "Value",
        "Lower Bound",
        "Upper Bound",
        "Explanation",
    ]

    return pd.DataFrame(anomalies, columns=result_columns)
