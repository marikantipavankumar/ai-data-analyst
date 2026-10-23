
import pandas as pd


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """
    Analyze a DataFrame for missing values, duplicate rows,
    data types, and potential data-quality issues.
    """

    if df.empty and len(df.columns) == 0:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "duplicate_rows": 0,
            "missing_values": pd.DataFrame(),
            "column_info": pd.DataFrame(),
        }

    # Count missing values in each column.
    missing_counts = df.isnull().sum()

    missing_values = pd.DataFrame({
        "Column": missing_counts.index,
        "Missing Count": missing_counts.values,
        "Missing Percentage": (
            missing_counts.values / len(df) * 100
            if len(df) > 0
            else 0
        ),
    })

    # Inspect the data type and number of unique values in each column.
    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Unique Values": [df[col].nunique() for col in df.columns],
    })

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": missing_values,
        "column_info": column_info,
    }
