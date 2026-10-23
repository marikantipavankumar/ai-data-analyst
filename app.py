import streamlit as st
import pandas as pd

from modules.data_quality import analyze_data_quality

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("AI-Powered Data Analyst")
st.write(
    "Upload your CSV files to explore your data "
    "and prepare for AI-powered analysis."
)

uploaded_files = st.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Dataset: {uploaded_file.name}")

        try:
            # Read the uploaded CSV file.
            df = pd.read_csv(uploaded_file)

            if len(df.columns) == 0:
                st.warning("No columns were found in this file.")
                continue

            if df.empty:
                st.warning("This CSV contains no data rows.")
                continue

            st.success("CSV uploaded and read successfully.")

            # Analyze data quality.
            quality = analyze_data_quality(df)

            # Display dataset metrics.
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Rows", quality["total_rows"])
            col2.metric("Columns", quality["total_columns"])
            col3.metric(
                "Missing Cells",
                int(df.isna().sum().sum()),
            )
            col4.metric(
                "Duplicate Rows",
                quality["duplicate_rows"],
            )

            # Display data preview.
            st.subheader("Data Preview")
            st.dataframe(
                df.head(100),
                use_container_width=True,
            )

            # Display column information.
            with st.expander("Column Information"):
                st.dataframe(
                    quality["column_info"],
                    use_container_width=True,
                )

            # Display missing values report.
            with st.expander("Missing Values Report"):
                st.dataframe(
                    quality["missing_values"],
                    use_container_width=True,
                )

        except (
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            st.error(f"Could not read this CSV: {exc}")

        except Exception as exc:
            st.error(
                f"Unexpected error processing "
                f"{uploaded_file.name}: {exc}"
            )

else:
    st.info("Upload a CSV file to get started.")
