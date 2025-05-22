import os
from dotenv import load_dotenv
from pathlib import Path
import re

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

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()


# Extract text from PDF
def extract_text_from_pdf(file) -> str:
    text_parts = []
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for page in doc:
            # Extract visible text
            text_parts.append(page.get_text())

            # Extract links separately
            links = page.get_links()
            for link in links:
                uri = link.get("uri")
                if uri:
                    text_parts.append(f"\n[Link]: {uri}")

    return "\n".join(text_parts)

# Extract text from DOCX
def extract_text_from_docx(file) -> str:
    doc = docx.Document(file.file)
    return " ".join([p.text for p in doc.paragraphs])

def preprocess_text(text: str) -> str:
    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[•\-\*\u2022]+", "-", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spacing
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()



def calculate_overall_confidence(data: Dict) -> float:
    scores = []

    for field in data.values():
        if isinstance(field, dict) and "confidence" in field:
            scores.append(field["confidence"])

    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 2)



# Prompt for GPT
def generate_prompt(text: str) -> str:
    return f'''
You are an intelligent resume parser. Extract the following fields from the resume text provided.

Fields to Extract:
- Full Name
- Email
- Phone Number
- LinkedIn URL 
- GitHub URL
- Location
- Summary
- Skills
- Education (Degree, Institution, Dates)
- Work Experience (Job Title, Company, Dates, Description)

Instructions:
- For each field, return an object with:
    - "value": the extracted content (string or list)
    - "confidence": a float between 0 and 1 representing your confidence in the accuracy of the value
- If a field is not found or you're unsure, set:
    - "value": null
    - "confidence": 0.0
- Do your best even with unstructured or messy formatting.

Return Format (example):
{{
  "full_name": {{ "value": "Jane Doe", "confidence": 0.98 }},
  "email": {{ "value": "jane@example.com", "confidence": 0.94 }},
  "skills": {{ "value": ["Python", "FastAPI"], "confidence": 0.89 }}
}}

Resume Text:
{text}
'''




# Call OpenAI and return structured JSON
def extract_data_with_gpt(text: str) -> Dict:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
        text = preprocess_text(text)
        extracted_data = extract_data_with_gpt(text)
        overall_score = calculate_overall_confidence(extracted_data)
        extracted_data["overall_confidence"] = overall_score
        return extracted_data

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optional root route
@app.get("/")
def root():
    return {"message": "Resume Parser is running. Visit /docs to test."}