import streamlit as st
import pandas as pd
import plotly.express as px

from modules.analysis_engine import answer_question
from modules.anomaly_detection import detect_anomalies
from modules.data_quality import analyze_data_quality
from modules.explanation import explain_analysis


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
            # Read and validate the uploaded CSV file.
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
            col3.metric("Missing Cells", int(df.isna().sum().sum()))
            col4.metric("Duplicate Rows", quality["duplicate_rows"])

            # Display data preview.
            st.subheader("Data Preview")
            st.dataframe(df.head(100), use_container_width=True)

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

            # Interactive data visualization.
            st.subheader("Data Visualization")

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            categorical_columns = [
                column
                for column in df.columns
                if not pd.api.types.is_numeric_dtype(df[column])
            ]

            chart_type = st.selectbox(
                "Choose a chart type",
                ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot"],
                key=f"chart_type_{uploaded_file.name}",
            )

            if not numeric_columns:
                st.info("No numeric columns are available for charting.")

            elif chart_type == "Scatter Plot":
                if len(numeric_columns) < 2:
                    st.info(
                        "A scatter plot requires at least two numeric columns."
                    )
                else:
                    col1, col2 = st.columns(2)

                    with col1:
                        x_column = st.selectbox(
                            "X-axis",
                            numeric_columns,
                            key=f"scatter_x_{uploaded_file.name}",
                        )

                    available_y_columns = [
                        column for column in numeric_columns if column != x_column
                    ]

                    with col2:
                        y_column = st.selectbox(
                            "Y-axis",
                            available_y_columns,
                            key=f"scatter_y_{uploaded_file.name}",
                        )

                    if st.button(
                        "Generate Chart",
                        key=f"scatter_generate_{uploaded_file.name}",
                    ):
                        chart_data = df[[x_column, y_column]].dropna()

                        if chart_data.empty:
                            st.warning("There is no valid data to chart.")
                        else:
                            fig = px.scatter(
                                chart_data,
                                x=x_column,
                                y=y_column,
                                title=f"{y_column} vs {x_column}",
                                hover_data=[x_column, y_column],
                            )
                            st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Pie Chart":
                if not categorical_columns:
                    st.info("A pie chart requires a categorical column.")
                else:
                    col1, col2 = st.columns(2)

                    with col1:
                        label_column = st.selectbox(
                            "Category column",
                            categorical_columns,
                            key=f"pie_label_{uploaded_file.name}",
                        )

                    with col2:
                        value_column = st.selectbox(
                            "Values column",
                            numeric_columns,
                            key=f"pie_value_{uploaded_file.name}",
                        )

                    if st.button(
                        "Generate Chart",
                        key=f"pie_generate_{uploaded_file.name}",
                    ):
                        pie_data = df[[label_column, value_column]].dropna()
                        pie_data = pie_data.groupby(
                            label_column,
                            as_index=False,
                        )[value_column].sum()

                        if pie_data.empty:
                            st.warning("There is no valid data to chart.")
                        else:
                            fig = px.pie(
                                pie_data,
                                names=label_column,
                                values=value_column,
                                title=f"{value_column} by {label_column}",
                            )
                            st.plotly_chart(fig, use_container_width=True)

            else:
                x_options = (
                    categorical_columns
                    if categorical_columns
                    else df.columns.tolist()
                )

                col1, col2 = st.columns(2)

                with col1:
                    x_column = st.selectbox(
                        "X-axis",
                        x_options,
                        key=f"x_axis_{uploaded_file.name}",
                    )

                with col2:
                    y_column = st.selectbox(
                        "Numeric column",
                        numeric_columns,
                        key=f"y_axis_{uploaded_file.name}",
                    )

                if st.button(
                    "Generate Chart",
                    key=f"bar_line_generate_{uploaded_file.name}",
                ):
                    chart_data = df[[x_column, y_column]].dropna()

                    if chart_data.empty:
                        st.warning("There is no valid data to chart.")
                    elif chart_type == "Bar Chart":
                        chart_data = chart_data.groupby(
                            x_column,
                            as_index=False,
                        )[y_column].sum()

                        fig = px.bar(
                            chart_data,
                            x=x_column,
                            y=y_column,
                            title=f"{y_column} by {x_column}",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Sort the X-axis when its values are valid dates.
                        parsed_dates = pd.to_datetime(
                            chart_data[x_column],
                            errors="coerce",
                        )

                        if parsed_dates.notna().all():
                            chart_data = chart_data.copy()
                            chart_data[x_column] = parsed_dates
                            chart_data = chart_data.sort_values(x_column)

                        fig = px.line(
                            chart_data,
                            x=x_column,
                            y=y_column,
                            title=f"{y_column} by {x_column}",
                            markers=True,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Detect unusual values in numeric columns.
            st.subheader("Anomaly Detection")
            anomalies = detect_anomalies(df)

            if anomalies.empty:
                st.success("No anomalies detected in numeric columns.")
            else:
                st.warning(f"Detected {len(anomalies)} potential anomalies.")
                st.dataframe(anomalies, use_container_width=True)

            # Ask questions about the dataset.
            st.subheader("Ask Your Data")
            question = st.text_input(
                "Enter your question",
                placeholder="e.g. What is the total revenue?",
                key=f"question_{uploaded_file.name}",
            )

            if st.button("Analyze", key=f"analyze_{uploaded_file.name}"):
                if question.strip():
                    answer = answer_question(df, question)

                    st.markdown("### Analysis Result")
                    st.text(answer)

                    # Explain the analysis result.
                    with st.expander("How was this answer calculated?"):
                        explanation = explain_analysis(
                            df=df,
                            question=question,
                        )

                        for line in explanation.splitlines():
                            label, separator, detail = line.partition(": ")

                            if separator:
                                st.markdown(f"**{label}:** {detail}")
                            else:
                                st.markdown(line)
                else:
                    st.warning("Please enter a question first.")

        except (
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            st.error(f"Could not read this CSV: {exc}")

        except Exception as exc:
            st.error(
                f"Unexpected error processing {uploaded_file.name}: {exc}"
            )

else:
    st.info("Upload a CSV file to get started.")
