import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Garage Service Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_port = int(os.getenv("MYSQL_PORT") or 3306)
db_name = os.getenv("MYSQL_DATABASE") or "service_management_db"

engine = create_engine(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
    pool_pre_ping=True
)

few_shot_examples = """
Q: List vehicles with their owner name and contact.
SQL: SELECT v.vehicle_id, v.VIN, v.license_plate, v.model, v.make, v.year,
            c.name AS customer_name, c.email, c.phone
     FROM vehicle_service_db.Vehicle AS v
     JOIN customer_service_db.Customer AS c ON v.customer_id = c.customer_id;

Q: Show upcoming appointments with customer and vehicle details.
SQL: SELECT a.appointment_id, a.appointment_date, a.time_slot, a.service_type, a.status,
            c.name AS customer_name, v.license_plate, v.model, v.make
     FROM appointment_service_db.Appointment AS a
     JOIN customer_service_db.Customer AS c ON a.customer_id = c.customer_id
     JOIN vehicle_service_db.Vehicle AS v ON a.vehicle_id = v.vehicle_id
     ORDER BY a.appointment_date ASC;

Q: List services with assigned employee and vehicle info.
SQL: SELECT s.service_id, s.service_type, s.status, s.start_time, s.end_time,
            e.name AS employee_name, v.license_plate, v.model
     FROM service_management_db.Service AS s
     JOIN employee_service_db.Employee AS e ON s.assigned_to = e.employee_id
     JOIN vehicle_service_db.Vehicle AS v ON s.vehicle_id = v.vehicle_id;

Q: Show updates for service 1 ordered by time.
SQL: SELECT u.update_id, u.progress_percentage, u.update_text, u.created_at
     FROM service_management_db.Service_Update AS u
     WHERE u.service_id = 1
     ORDER BY u.created_at ASC;

Q: Total time logged per employee.
SQL: SELECT e.employee_id, e.name, SUM(t.duration_minutes) AS total_minutes
     FROM employee_service_db.Employee AS e
     JOIN employee_service_db.Time_Log AS t ON e.employee_id = t.employee_id
     GROUP BY e.employee_id, e.name
     ORDER BY total_minutes DESC;
"""

def generate_sql(question: str) -> str:
    prompt = f"""
You are an expert MySQL SQL generator for a garage service microservices database.

Databases and tables:
- customer_service_db.Customer (customer_id, first_name, last_name, name, email, phone, password_hash, created_at)
- employee_service_db.Employee (employee_id, first_name, last_name, name, email, password_hash, role, photo_url, status, hourly_rate, specialization, hire_date, created_at)
- employee_service_db.Time_Log (log_id, employee_id, work_type, start_time, end_time, duration_minutes, description, created_at)
- vehicle_service_db.Vehicle (vehicle_id, customer_id, VIN, license_plate, model, make, year, color, mileage, status, updated_at)
- appointment_service_db.Appointment (appointment_id, customer_id, vehicle_id, appointment_date, time_slot, service_type, status, created_at, updated_at)
- service_management_db.Service (service_id, vehicle_id, assigned_to, service_type, description, start_time, end_time, estimated_cost, actual_cost, completion_percentage, notes, status, created_at, updated_at)
- service_management_db.Service_Update (update_id, service_id, progress_percentage, update_text, created_at)

Relationships:
- Vehicle.customer_id → Customer.customer_id
- Appointment.customer_id → Customer.customer_id; Appointment.vehicle_id → Vehicle.vehicle_id
- Service.vehicle_id → Vehicle.vehicle_id; Service.assigned_to → Employee.employee_id
- Service_Update.service_id → Service.service_id

Requirements:
- ALWAYS fully-qualify tables with their database name (e.g., service_management_db.Service).
- Generate ONE valid MySQL SELECT statement only. No comments, no markdown, no DDL/DML.
- Prefer JOINs to combine related entities according to the relationships above.
- Include ORDER BY when the question implies sorting (e.g., latest, top, upcoming).

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
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        return str(e)


def explain_results(question: str, df):
    if isinstance(df, str):
        return f"I couldn’t process your question because of an error: {df}"

    if df.empty:
        return "I couldn’t find any matching records for that question."

    data_str = df.to_string(index=False)

    prompt = f"""
You are a friendly garage service assistant.
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

class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: QuestionRequest):
    question = request.question
    sql_query = generate_sql(question)
    result_df = execute_sql(sql_query)
    reply = explain_results(question, result_df)
    return {"answer": reply}

@app.get("/")
async def root():
    return {"message": "Garage Service Chatbot API is running 🚀"}
