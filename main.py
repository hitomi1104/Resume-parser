import os
from dotenv import load_dotenv
from pathlib import Path

# ✅ Force load the .env from current file location
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

# Debug: Print API key (remove this in production)
print("DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict
import fitz  # PyMuPDF for PDF parsing
import docx
import json
from openai import OpenAI

# ✅ Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

# ✅ Initialize OpenAI client
client = OpenAI(api_key=api_key)

# ✅ Initialize FastAPI app
app = FastAPI()

# Extract text from PDF
def extract_text_from_pdf(file) -> str:
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        return " ".join([page.get_text() for page in doc])

# Extract text from DOCX
def extract_text_from_docx(file) -> str:
    doc = docx.Document(file.file)
    return " ".join([p.text for p in doc.paragraphs])

# Prompt for GPT
def generate_prompt(text: str) -> str:
    return f"""
Extract the following fields from this resume:
- Full Name
- Email
- Phone Number
- LinkedIn URL
- Location
- Summary
- Skills
- Education (Degree, Institution, Dates)
- Work Experience (Job Title, Company, Dates, Description)

Text:
{text}

Return the result as valid JSON.
"""

# Call OpenAI and return structured JSON
def extract_data_with_gpt(text: str) -> Dict:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": generate_prompt(text)}],
        temperature=0.2,
    )
    raw_output = response.choices[0].message.content
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"raw_output": raw_output}  # fallback if JSON is invalid

# API endpoint
@app.post("/parse")
async def parse_resume(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(file)
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    try:
        extracted_data = extract_data_with_gpt(text)
        return extracted_data
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optional root route
@app.get("/")
def root():
    return {"message": "Resume Parser is running. Visit /docs to test."}