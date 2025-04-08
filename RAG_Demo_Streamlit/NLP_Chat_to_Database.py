import os
import streamlit as st
import boto3
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

# Load environment variables from .env
load_dotenv()

# Sidebar for user inputs
#st.sidebar.image("logo.png", use_container_width=True)  # Add your logo file in the same directory as the script
st.sidebar.title("Configuration")

# Prompt user for sensitive keys in the sidebar
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
aws_secret_access_key = st.sidebar.text_input("Enter your AWS Secret Access Key", type="password")
database_name = st.sidebar.text_input("Enter Athena Database Name", placeholder="e.g., my_database")

# Load other keys from .env
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_region = os.getenv("AWS_REGION", "us-east-1")
aws_staging_dir = os.getenv("AWS_S3_STAGING_DIR")

# Ensure required keys are provided
if not openai_api_key or not aws_secret_access_key or not database_name:
    st.sidebar.warning("Please provide OpenAI API Key, AWS Secret Access Key, and Athena Database Name.")
    st.stop()

# Initialize Athena client
athena_client = boto3.client(
    "athena",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)

# Streamlit app layout
st.title("Chat with AWS Athena Tables")

# User input for natural language question
user_question = st.text_area("Ask a question in natural language", placeholder="e.g., How many rows are in the table?")

# Function to fetch schema information from Athena
def fetch_schema(database_name):
    try:
        query = f"""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{database_name}'
        """
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": database_name},
            ResultConfiguration={"OutputLocation": "s3://awsbucketforagents/"},
        )
        query_execution_id = response["QueryExecutionId"]

        # Wait for the query to complete with a timeout
        status = "RUNNING"
        start_time = time.time()
        timeout = 60  # Timeout in seconds
        while status in ["RUNNING", "QUEUED"]:
            if time.time() - start_time > timeout:
                st.error("Query timed out. Please try again later.")
                return None
            response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            status = response["QueryExecution"]["Status"]["State"]
            st.info(f"Query status: {status}")
            time.sleep(2)  # Poll every 2 seconds

        if status == "SUCCEEDED":
            # Fetch results
            result = athena_client.get_query_results(QueryExecutionId=query_execution_id)
            rows = result["ResultSet"]["Rows"]
            schema = {}
            for row in rows[1:]:  # Skip header row
                table_name = row["Data"][0]["VarCharValue"]
                column_name = row["Data"][1]["VarCharValue"]
                data_type = row["Data"][2]["VarCharValue"]
                if table_name not in schema:
                    schema[table_name] = []
                schema[table_name].append((column_name, data_type))
            return schema
        else:
            st.error(f"Failed to fetch schema with status: {status}")
            return None
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return None

# Function to generate SQL query from natural language
def generate_sql_query(question, database_name, schema):
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    schema_description = "\n".join(
        [f"Table: {table}\nColumns: {', '.join([f'{col[0]} ({col[1]})' for col in columns])}" for table, columns in schema.items()]
    )
    prompt_template = """
    You are an AI assistant that generates SQL queries for AWS Athena. The user will provide a natural language question, and you will generate a valid SQL query for the given database.

    Database: {database_name}
    Schema:
    {schema_description}

    Question: {question}
    SQL Query:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["database_name", "schema_description", "question"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    sql_query = llm_chain.run({"database_name": database_name, "schema_description": schema_description, "question": question})
    sql_query = sql_query.strip("```sql").strip("```").strip()

    # Validate the generated SQL query
    valid_sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "DESCRIBE", "SHOW"]
    if not any(sql_query.upper().startswith(keyword) for keyword in valid_sql_keywords):
        # If the query is invalid, return it as a result instead of executing it
        return sql_query, False

    return sql_query, True

# Function to execute query in Athena
def execute_athena_query(query, database):
    try:
        st.write(f"Executing SQL Query: `{query}`")  # Log the query being executed
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": database},
            ResultConfiguration={"OutputLocation": "s3://awsbucketforagents/"},
        )
        query_execution_id = response["QueryExecutionId"]

        # Wait for the query to complete
        status = "RUNNING"
        while status in ["RUNNING", "QUEUED"]:
            response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            status = response["QueryExecution"]["Status"]["State"]
            st.info(f"Query status: {status}")
            time.sleep(2)  # Poll every 2 seconds

        if status == "SUCCEEDED":
            # Fetch results
            result = athena_client.get_query_results(QueryExecutionId=query_execution_id)
            rows = result["ResultSet"]["Rows"]
            columns = [col["VarCharValue"] for col in rows[0]["Data"]]
            data = [[col.get("VarCharValue", None) for col in row["Data"]] for row in rows[1:]]
            return pd.DataFrame(data, columns=columns)
        else:
            st.error(f"Query failed with status: {status}")
            return None
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None

# Process user question
if st.button("Ask Question"):
    if user_question:
        st.info("Fetching schema...")
        schema = fetch_schema(database_name)
        if schema:
            st.info("Generating SQL query...")
            sql_query, is_valid = generate_sql_query(user_question, database_name, schema)
            st.write(f"Generated SQL Query: `{sql_query}`")

            if is_valid:
                st.info("Running query...")
                df = execute_athena_query(sql_query, database_name)
                if df is not None:
                    st.write("Query Results:")
                    st.dataframe(df)
            else:
                st.warning("The question is too general or unrelated to the database. Displaying the generated SQL query instead.")
    else:
        st.warning("Please provide a question.")
