import os
from dotenv import load_dotenv
from pathlib import Path
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict, List
import fitz  # PyMuPDF
import docx
import json
from openai import OpenAI
from PIL import Image
import io
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import requests

# Load .env
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

# OpenAI Setup
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")
client = OpenAI(api_key=api_key)

# LandingAI Setup
landingai_api_key = os.getenv("LANDINGAI_API_KEY")

# FastAPI app
app = FastAPI()

# -------- Extractors --------
def extract_text_from_pdf(file) -> str:
    text_parts = []
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
            links = page.get_links()
            for link in links:
                uri = link.get("uri")
                if uri:
                    text_parts.append(f"\n[Link]: {uri}")
    return "\n".join(text_parts)

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file.file)
    return " ".join([p.text for p in doc.paragraphs])

def extract_text_from_linkedin(url: str) -> str:
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(5)
        text = driver.find_element(By.TAG_NAME, 'body').text
        driver.quit()
        return text
    except Exception as e:
        return f"Error scraping LinkedIn: {e}"

# NEW: Use LandingAI for OCR
def extract_with_landingai(images: List[UploadFile]) -> Dict:
    try:
        headers = {"Authorization": f"Bearer {landingai_api_key}"}
        files = [("files", (img.filename, img.file, img.content_type)) for img in images]
        response = requests.post(
            "https://api.landing.ai/v2/inference/resume-parser",
            headers=headers,
            files=files
        )
        return response.json()
    except Exception as e:
        return {"error": f"LandingAI OCR failed: {str(e)}"}

# -------- Utilities --------
def preprocess_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[•\-\*\u2022]+", "-", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()

def generate_prompt(text: str) -> str:
    return f'''
You are an intelligent resume parser. Extract the following fields:
- Full Name, Email, Phone Number, LinkedIn URL, GitHub URL, Location, Summary, Skills, Education, Work Experience.
Return each as an object with "value" and "confidence".
Resume Text:
{text}'''

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
        return {"raw_output": raw_output}

def calculate_overall_confidence(data: Dict) -> float:
    scores = [v["confidence"] for v in data.values() if isinstance(v, dict) and "confidence" in v]
    return round(sum(scores) / len(scores), 2) if scores else 0.0

# -------- Routes --------
@app.post("/parse")
async def parse_resume(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)
        processed = preprocess_text(text)
        data = extract_data_with_gpt(processed)
        data["overall_confidence"] = calculate_overall_confidence(data)
        return data
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/parse/Linkedin_url")
async def parse_linkedin(url: str):
    try:
        text = extract_text_from_linkedin(url)
        processed = preprocess_text(text)
        data = extract_data_with_gpt(processed)
        data["overall_confidence"] = calculate_overall_confidence(data)
        return data
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/parse/images")
async def parse_images_with_landingai(files: List[UploadFile] = File(...)):
    result = extract_with_landingai(files)
    return result

@app.get("/")
def root():
    return {"message": "Resume Parser is running. Visit /docs."} 