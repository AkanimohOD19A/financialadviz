import nest_asyncio

nest_asyncio.apply()

import os
import base64
import gc
import random
import tempfile
import logging
import time
import uuid
import pandas as pd
import numpy as np
import asyncio
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core import PromptTemplate
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Financial Intelligence Advisor", page_icon="💰", layout="wide")
st.title("Financial Intelligence Advisor")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.dataframes = {}
    st.session_state.current_analysis = None

session_id = st.session_state.id


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file_path):
    # Opening file from file path
    st.markdown("### PDF Preview")
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    # Embedding PDF in HTML
    pdf_display = f"""<iframe
    src="data:application/pdf;base64,{base64_pdf}" width="400" height="400"
    type="application/pdf">
    </iframe>"""
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_csv(file_path):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            logger.info(f"Attempting to load CSV with {encoding} encoding")
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            return df
        except UnicodeDecodeError as ue:
            logger.info(f"Failed with {encoding}: {ue} - Trying next encoding")
            continue
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise


def display_dataframe(df, max_rows=10):
    with st.expander("Preview Data"):
        st.dataframe(df.head(max_rows))
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("Column Types:")
        for col, dtype in df.dtypes.items():
            st.write(f"- {col}: {dtype}")


class FinancialDataAnalyzer:
    """Helper class to perform financial data analysis and aggregations"""

    @staticmethod
    def get_summary_stats(df):
        """Get basic summary statistics for numerical columns"""
        return df.describe()

    @staticmethod
    def get_correlation_matrix(df):
        """Get correlation matrix for numerical columns"""
        numeric_df = df.select_dtypes(include=['number'])
        return numeric_df.corr()

    @staticmethod
    def perform_aggregation(df, group_by_cols, agg_cols, agg_funcs=['mean', 'sum', 'count']):
        """Perform aggregation on dataframe"""
        if not all(col in df.columns for col in group_by_cols + agg_cols):
            missing = [col for col in group_by_cols + agg_cols if col not in df.columns]
            return f"Error: Columns {missing} not found in dataframe"

        agg_dict = {col: agg_funcs for col in agg_cols}
        result = df.groupby(group_by_cols).agg(agg_dict)
        return result

    @staticmethod
    def generate_time_series(df, date_col, value_col, freq='M'):
        """Generate time series analysis"""
        if date_col not in df.columns or value_col not in df.columns:
            return f"Error: Required columns not found in dataframe"

        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                return f"Error: Could not convert {date_col} to datetime"

        # Resample data
        df_sorted = df.sort_values(by=date_col)
        df_sorted = df_sorted.set_index(date_col)
        resampled = df_sorted[value_col].resample(freq).sum()

        return resampled.reset_index()

    @staticmethod
    def calculate_financial_metrics(df, amount_col):
        """Calculate common financial metrics"""
        if amount_col not in df.columns:
            return f"Error: {amount_col} column not found in dataframe"

        metrics = {}
        metrics['total'] = df[amount_col].sum()
        metrics['average'] = df[amount_col].mean()
        metrics['median'] = df[amount_col].median()
        metrics['std_dev'] = df[amount_col].std()
        metrics['min'] = df[amount_col].min()
        metrics['max'] = df[amount_col].max()
        metrics['range'] = metrics['max'] - metrics['min']

        # Calculate quartiles
        metrics['q1'] = df[amount_col].quantile(0.25)
        metrics['q3'] = df[amount_col].quantile(0.75)
        metrics['iqr'] = metrics['q3'] - metrics['q1']

        return metrics


def execute_financial_query(query_str, dataframes):
    """Execute a financial query using pandas operations"""
    try:
        # Create a safe locals dictionary with only the dataframes
        safe_locals = {f"df{i}": df for i, df in enumerate(dataframes.values())}
        safe_locals.update({
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'analyzer': FinancialDataAnalyzer
        })

        # Execute the query in the safe environment
        result = eval(query_str, {"__builtins__": {}}, safe_locals)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"


def create_financial_assistant(api_key, docs, dataframes):
    """Create a financial assistant with RAG capabilities"""
    llm = Cohere(api_key=api_key, model="command-r-plus")
    embed_model = CohereEmbedding(
        cohere_api_key=api_key,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )

    cohere_rerank = CohereRerank(
        model='rerank-english-v3.0',
        api_key=api_key,
    )

    # Configure global settings
    Settings.embed_model = embed_model
    Settings.llm = llm

    # Create document index
    doc_index = VectorStoreIndex.from_documents(docs, show_progress=True)

    # Create a query engine for documents
    doc_query_engine = doc_index.as_query_engine(
        streaming=True,
        node_postprocessors=[cohere_rerank]
    )

    # Create a custom financial prompt template
    financial_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "You are a financial advisor assistant specializing in data analysis and providing financial insights. "
        "You have access to financial datasets and reports. "
        "When responding to queries:\n"
        "1. Analyze the financial data carefully and provide insights that would be valuable to a financial service provider\n"
        "2. When appropriate, suggest potential actions or strategies based on the data\n"
        "3. If you're uncertain about specific details, acknowledge limitations in your analysis\n"
        "4. Keep responses structured and concise, with clear financial terminology\n"
        "5. If calculations or aggregations are needed, explain your methodology\n\n"
        "Think step by step to answer the query in a professional, analytical manner.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    financial_prompt_tmpl = PromptTemplate(financial_prompt_tmpl_str)
    doc_query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": financial_prompt_tmpl}
    )

    # Create tools for each dataframe
    tools = []

    # Add document query engine as a tool
    tools.append(
        QueryEngineTool(
            query_engine=doc_query_engine,
            metadata=ToolMetadata(
                name="document_tool",
                description="Useful for answering questions about financial reports and documents"
            )
        )
    )

    # Add dataframe tools if available
    for name, df in dataframes.items():
        # Create a string representation of the dataframe schema
        schema_str = f"Dataframe '{name}' with {df.shape[0]} rows and {df.shape[1]} columns.\n"
        schema_str += "Columns:\n"
        for col, dtype in df.dtypes.items():
            schema_str += f"- {col}: {dtype}\n"

        # Add sample data
        schema_str += f"\nSample data (first 5 rows):\n{df.head(5).to_string()}\n"

        # Create an in-memory document for the dataframe schema
        from llama_index.core.schema import Document
        schema_doc = Document(text=schema_str)

        # Create an index for the schema
        schema_index = VectorStoreIndex.from_documents([schema_doc])

        # Create a query engine for the schema
        schema_query_engine = schema_index.as_query_engine(
            streaming=True,
            node_postprocessors=[cohere_rerank]
        )

        # Add the schema tool
        tools.append(
            QueryEngineTool(
                query_engine=schema_query_engine,
                metadata=ToolMetadata(
                    name=f"{name}_tool",
                    description=f"Useful for answering questions about the '{name}' financial dataset"
                )
            )
        )

    # Create a direct query engine first (as fallback)
    if docs:
        direct_engine = doc_query_engine
    elif dataframes:
        # Use the first dataframe's query engine as fallback if no docs
        direct_engine = tools[0].query_engine
    else:
        # This is just a placeholder in case neither docs nor dataframes are provided
        from llama_index.core.schema import Document
        empty_doc = Document(text="No documents or datasets provided.")
        empty_index = VectorStoreIndex.from_documents([empty_doc])
        direct_engine = empty_index.as_query_engine()

    # Try creating the sub-question query engine with proper error handling
    try:
        from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
        from llama_index.core.question_gen.types import SubQuestion

        # Create a more reliable question generator with custom prompt
        SUB_QUESTION_PROMPT_TMPL = """You are an AI assistant tasked with decomposing a complex financial query into simpler sub-questions.
        For the given query, generate specific sub-questions that would help answer the overall query.
        Each sub-question should be self-contained and focused on a specific aspect of the query. 
        Provide at most 3 sub-questions.

        Original query: {query_str}

        Generate sub-questions in the following format:
        1. [sub-question 1]
        2. [sub-question 2]
        3. [sub-question 3]
        """

        sub_question_prompt = PromptTemplate(SUB_QUESTION_PROMPT_TMPL)

        # Create a proper question generator
        question_gen = LLMQuestionGenerator.from_defaults(
            llm=llm,
            prompt=sub_question_prompt,
            verbose=True
        )

        # Create the sub-question query engine with proper configuration
        query_engine = SubQuestionQueryEngine.from_defaults(
            question_gen=question_gen,
            query_engine_tools=tools,
            llm=llm,
            verbose=True,
            use_async=False,  # Set to False to avoid async issues
            # Add a response synthesizer to properly combine the sub-question responses
            response_synthesizer_kwargs={
                "response_mode": "compact",
                "verbose": True
            }
        )

        # Test the engine with a simple query to catch errors early
        try:
            logger.info("Testing SubQuestionQueryEngine with a simple query")
            test_response = query_engine.query("Test")
            logger.info("SubQuestionQueryEngine test successful")
        except Exception as test_e:
            logger.error(f"SubQuestionQueryEngine test failed: {test_e}")
            # If the SubQuestionQueryEngine fails, fall back to direct engine
            logger.warning("Falling back to direct query engine")
            query_engine = direct_engine

    except Exception as e:
        # If there's any issue setting up SubQuestionQueryEngine, fall back to direct engine
        logger.error(f"Error setting up SubQuestionQueryEngine: {e}")
        query_engine = direct_engine

    return query_engine


# Predefined file paths and API key
API_KEY = st.secrets["COHERE_API_KEY"]  # Fetch API key from environment variable
PDF_FILE_PATH = "docs/Re_Data-Analyst-Take-Home-Assessment-_Daniel-AMAH.pdf"  # Replace with your PDF file path
CSV_FILE_PATH1 = "docs/cleaned_uesr_table.csv"  # Replace with your first CSV file path
CSV_FILE_PATH2 = "docs/cleaned_transactions_table.csv"  # Replace with your second CSV file path

# Load files internally
try:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Store uploaded files
        uploaded_files = []
        dataframes = {}

        # Process PDF file
        if os.path.exists(PDF_FILE_PATH):
            uploaded_files.append(PDF_FILE_PATH)
            st.sidebar.success(f"Processed PDF: {PDF_FILE_PATH}")

        # Process CSV files
        if os.path.exists(CSV_FILE_PATH1):
            try:
                df1 = load_csv(CSV_FILE_PATH1)
                dataframes["dataset1"] = df1
                st.session_state.dataframes["dataset1"] = df1
                st.sidebar.success(f"Loaded Dataset 1: {CSV_FILE_PATH1} ({df1.shape[0]} rows)")
            except Exception as e:
                st.sidebar.error(f"Error loading Dataset 1: {e}")

        if os.path.exists(CSV_FILE_PATH2):
            try:
                df2 = load_csv(CSV_FILE_PATH2)
                dataframes["dataset2"] = df2
                st.session_state.dataframes["dataset2"] = df2
                st.sidebar.success(f"Loaded Dataset 2: {CSV_FILE_PATH2} ({df2.shape[0]} rows)")
            except Exception as e:
                st.sidebar.error(f"Error loading Dataset 2: {e}")

        # Load documents if files were uploaded
        if uploaded_files:
            st.sidebar.info("Indexing documents...")
            loader = SimpleDirectoryReader(
                input_files=uploaded_files,
                required_exts=[".pdf"],
                recursive=True
            )
            docs = loader.load_data()
        else:
            docs = []

        # Create financial assistant
        st.sidebar.info("Creating financial assistant...")
        query_engine = create_financial_assistant(API_KEY, docs, dataframes)

        # Store the query engine in session state
        cache_key = f"{session_id}-financial-assistant"
        st.session_state.file_cache[cache_key] = query_engine

        # Inform the user that processing is complete
        st.sidebar.success("Financial Assistant Ready!")

        # Preview files if available
        if os.path.exists(PDF_FILE_PATH):
            with st.sidebar.expander("PDF Preview"):
                display_pdf(PDF_FILE_PATH)

        # Preview dataframes in main area
        if dataframes:
            st.subheader("Dataset Previews")
            tabs = st.tabs([f"Dataset {i + 1}" for i in range(len(dataframes))])
            for i, (name, df) in enumerate(dataframes.items()):
                with tabs[i]:
                    display_dataframe(df)

except Exception as e:
    st.sidebar.error(f"An error occurred: {e}")
    logger.error(f"Error during processing: {e}", exc_info=True)

# Data Analysis Tools Section
st.sidebar.header("Data Analysis Tools")

if st.session_state.dataframes:
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        list(st.session_state.dataframes.keys())
    )

    if dataset_choice:
        df = st.session_state.dataframes[dataset_choice]

        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            ["Summary Statistics", "Correlation Matrix", "Aggregation", "Time Series", "Financial Metrics"]
        )

        if analysis_type == "Summary Statistics":
            if st.sidebar.button("Generate Summary"):
                stats = FinancialDataAnalyzer.get_summary_stats(df)
                st.session_state.current_analysis = {
                    "type": "summary",
                    "data": stats,
                    "title": f"Summary Statistics for {dataset_choice}"
                }

        elif analysis_type == "Correlation Matrix":
            if st.sidebar.button("Generate Correlation Matrix"):
                corr = FinancialDataAnalyzer.get_correlation_matrix(df)
                st.session_state.current_analysis = {
                    "type": "correlation",
                    "data": corr,
                    "title": f"Correlation Matrix for {dataset_choice}"
                }

        elif analysis_type == "Aggregation":
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()

            group_cols = st.sidebar.multiselect("Group By", cat_cols)
            agg_cols = st.sidebar.multiselect("Aggregate Columns", num_cols)

            if group_cols and agg_cols:
                if st.sidebar.button("Generate Aggregation"):
                    result = FinancialDataAnalyzer.perform_aggregation(df, group_cols, agg_cols)
                    st.session_state.current_analysis = {
                        "type": "aggregation",
                        "data": result,
                        "title": f"Aggregation for {dataset_choice}"
                    }

        elif analysis_type == "Time Series":
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            value_cols = df.select_dtypes(include=['number']).columns.tolist()

            date_col = st.sidebar.selectbox("Date Column", date_cols if date_cols else df.columns.tolist())
            value_col = st.sidebar.selectbox("Value Column", value_cols)

            if date_col and value_col:
                if st.sidebar.button("Generate Time Series"):
                    result = FinancialDataAnalyzer.generate_time_series(df, date_col, value_col)
                    st.session_state.current_analysis = {
                        "type": "time_series",
                        "data": result,
                        "date_col": date_col,
                        "value_col": value_col,
                        "title": f"Time Series of {value_col} by {date_col}"
                    }

        elif analysis_type == "Financial Metrics":
            amount_cols = [col for col in df.select_dtypes(include=['number']).columns
                           if any(
                    term in col.lower() for term in ['amount', 'price', 'cost', 'revenue', 'income', 'sale'])]

            amount_col = st.sidebar.selectbox(
                "Amount Column",
                amount_cols if amount_cols else df.select_dtypes(include=['number']).columns.tolist()
            )

            if amount_col:
                if st.sidebar.button("Calculate Financial Metrics"):
                    metrics = FinancialDataAnalyzer.calculate_financial_metrics(df, amount_col)
                    st.session_state.current_analysis = {
                        "type": "financial_metrics",
                        "data": metrics,
                        "column": amount_col,
                        "title": f"Financial Metrics for {amount_col}"
                    }

# Main chat interface
col1, col2 = st.columns([6, 1])
with col1:
    st.header("Financial Advisor Chat")
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Display current analysis if available
if st.session_state.current_analysis:
    analysis = st.session_state.current_analysis
    st.subheader(analysis["title"])

    if analysis["type"] == "summary":
        st.dataframe(analysis["data"])

    elif analysis["type"] == "correlation":
        fig = px.imshow(
            analysis["data"],
            text_auto=True,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif analysis["type"] == "aggregation":
        st.dataframe(analysis["data"])

    elif analysis["type"] == "time_series":
        if isinstance(analysis["data"], pd.DataFrame):
            fig = px.line(
                analysis["data"],
                x=analysis["date_col"],
                y=analysis["value_col"],
                title=f"Time Series of {analysis['value_col']}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(analysis["data"])  # Display error message

    elif analysis["type"] == "financial_metrics":
        metrics = analysis["data"]
        if isinstance(metrics, dict):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total", f"{metrics['total']:,.2f}")
                st.metric("Average", f"{metrics['average']:,.2f}")
                st.metric("Median", f"{metrics['median']:,.2f}")
                st.metric("Standard Deviation", f"{metrics['std_dev']:,.2f}")

            with col2:
                st.metric("Minimum", f"{metrics['min']:,.2f}")
                st.metric("Maximum", f"{metrics['max']:,.2f}")
                st.metric("Range", f"{metrics['range']:,.2f}")
                st.metric("IQR", f"{metrics['iqr']:,.2f}")

            # Create a box plot
            fig = go.Figure()
            fig.add_trace(go.Box(
                q1=[metrics['q1']], median=[metrics['median']],
                q3=[metrics['q3']], mean=[metrics['average']],
                lowerfence=[metrics['min']], upperfence=[metrics['max']],
                y=[analysis["column"]],
                name=analysis["column"],
                orientation="h"
            ))
            fig.update_layout(title=f"Distribution of {analysis['column']}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(metrics)  # Display error message

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about your financial data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if a query engine is available
    cache_key = f"{session_id}-financial-assistant"
    if cache_key in st.session_state.file_cache:
        query_engine = st.session_state.file_cache[cache_key]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Get the response
                response = query_engine.query(prompt)

                # Check if response needs async handling
                if hasattr(response, 'response_gen'):
                    # For streaming response
                    if hasattr(response.response_gen, '__aiter__'):
                        # Need to run async code - use nest_asyncio to handle this
                        async def process_async_stream(response_gen):
                            result = ""
                            async for chunk in response_gen:
                                result += chunk
                                message_placeholder.markdown(result + "▌")
                            return result


                        # Run the async function using asyncio
                        full_response = asyncio.run(process_async_stream(response.response_gen))
                    elif hasattr(response.response_gen, '__iter__'):
                        # For synchronous streaming response
                        for chunk in response.response_gen:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    else:
                        # Fall back to just using the response string
                        full_response = str(response)
                else:
                    # For non-streaming response
                    full_response = str(response)

                message_placeholder.markdown(full_response)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                message_placeholder.markdown(error_msg)
                full_response = error_msg
                logger.error(f"Query error: {str(e)}", exc_info=True)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        # No query engine available - prompt user to process files
        with st.chat_message("assistant"):
            st.markdown("Please upload files and process them first using the sidebar options.")

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please upload files and process them first using the sidebar options."
        })