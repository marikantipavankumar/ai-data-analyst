
import streamlit as st
import pandas as pd

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
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.warning("This CSV contains no data rows.")
                continue

            if len(df.columns) == 0:
                st.warning("No columns were found in this file.")
                continue

            st.success("CSV uploaded and read successfully.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric(
                "Missing cells",
                int(df.isna().sum().sum()),
            )

            st.subheader("Data Preview")
            st.dataframe(df.head(100), use_container_width=True)

            with st.expander("Column Information"):
                info = pd.DataFrame({
                    "Column": df.columns.astype(str),
                    "Data Type": df.dtypes.astype(str).values,
                    "Missing Values": df.isna().sum().values,
                    "Unique Values": df.nunique().values,
                })
                st.dataframe(info, use_container_width=True)

        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not read this CSV: {exc}")

        except Exception as exc:
            st.error(f"Unexpected error reading the file: {exc}")
else:
    st.info("Upload a CSV file to get started.")
