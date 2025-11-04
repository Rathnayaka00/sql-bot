import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Initialize FastAPI ---
app = FastAPI(title="Retail Sales Chatbot API")

# Allow frontend (React, Streamlit, etc.) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Database setup ---
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_name = os.getenv("MYSQL_DATABASE")

engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# --- Few-shot examples ---
few_shot_examples = """
Q: Show me the total sales for each product category.
SQL: SELECT ProductCategory, SUM(TotalAmount) AS TotalSales FROM sales_tb GROUP BY ProductCategory;

Q: List all customers who spent more than 500.
SQL: SELECT CustomerID, SUM(TotalAmount) AS TotalSpent FROM sales_tb GROUP BY CustomerID HAVING TotalSpent > 500;

Q: Show the top 5 product categories by total sales.
SQL: SELECT ProductCategory, SUM(TotalAmount) AS TotalSales FROM sales_tb GROUP BY ProductCategory ORDER BY TotalSales DESC LIMIT 5;

Q: Find the average age of customers by gender.
SQL: SELECT Gender, AVG(Age) AS AverageAge FROM sales_tb GROUP BY Gender;
"""


# --- Helper Functions ---
def generate_sql(question: str) -> str:
    """Generate SQL query from natural language."""
    prompt = f"""
You are an expert MySQL SQL generator. 
Database: 'retail_sales_db' with table 'sales_tb' (TransactionID, Date, CustomerID, Gender, Age, ProductCategory, Quantity, PriceperUnit, TotalAmount).

Given a question, write a valid MySQL query.
Return only the SQL statement, no explanations or formatting.

Examples:
{few_shot_examples}

Q: {question}
SQL:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    sql_query = response.choices[0].message.content.strip()
    return sql_query.replace("```sql", "").replace("```", "").strip()


def execute_sql(query: str):
    """Execute SQL query and return pandas DataFrame."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        return str(e)


def explain_results(question: str, df):
    """Generate natural language summary of results."""
    if isinstance(df, str):
        return f"⚠️ I couldn’t process your question because of an error: {df}"

    if df.empty:
        return "I couldn’t find any matching records for that question."

    data_str = df.to_string(index=False)

    prompt = f"""
You are a friendly retail sales assistant.
The user asked: "{question}"

Here are the query results:
{data_str}

Write a short and natural explanation in plain language — no SQL, just summarize like you’re talking to the user.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


# --- Request Body Schema ---
class QuestionRequest(BaseModel):
    question: str


# --- API Endpoint ---
@app.post("/chat")
async def chat(request: QuestionRequest):
    """Main chatbot endpoint."""
    question = request.question
    sql_query = generate_sql(question)
    result_df = execute_sql(sql_query)
    reply = explain_results(question, result_df)
    return {"answer": reply}


# --- Root route ---
@app.get("/")
async def root():
    return {"message": "Retail Sales Chatbot API is running 🚀"}
