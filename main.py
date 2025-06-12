import os
from dotenv import load_dotenv
from pathlib import Path
import re
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict
import fitz  # PyMuPDF for PDF parsing
import docx
import json
import openai
from openai import OpenAI
from rapidfuzz import process, fuzz
from fastapi import Query
import pandas as pd

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

client = OpenAI(api_key=api_key)
app = FastAPI()

def get_embedding(text: str, model="text-embedding-3-small") -> list:
    response = openai.embeddings.create(model=model, input=text)
    return response.data[0].embedding

####################################### Reading CSV and compare similarities with Docs #######################################

# Load SNOMED data
df = pd.read_csv("SNOMED_mappings_scored.csv", sep=";", index_col=False)
df = df.dropna(subset=["Dx", "SNOMED CT Code"])

# Generate or load OpenAI embeddings
if os.path.exists("snomed_with_embeddings.pkl"):
    print("✅ Loading precomputed SNOMED embeddings...")
    df = pd.read_pickle("snomed_with_embeddings.pkl")
else:
    print("⏳ Generating embeddings for SNOMED Dx terms. This may take a while...")
    df["embedding"] = df["Dx"].apply(lambda x: get_embedding(x))
    df.to_pickle("snomed_with_embeddings.pkl")

# Cosine similarity
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Matching using OpenAI embeddings
def match_indication_to_snomed(indication: str, top_n: int = 3):
    input_emb = get_embedding(indication)
    df_copy = df.copy()
    df_copy["similarity"] = df_copy["embedding"].apply(lambda e: cosine_similarity(e, input_emb))
    top_matches = df_copy.sort_values(by="similarity", ascending=False).head(top_n)

    results = []
    for _, row in top_matches.iterrows():
        results.append({
            "similarity_score": round(row["similarity"], 4),
            "indication": row["Dx"],
            "snomed_code": row["SNOMED CT Code"],
            "abbreviation": row.get("Abbreviation", None)
        })

    return results


####################################### Extractors #######################################
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


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

def extract_text_from_linkedin(url: str) -> str:
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=options)

        driver.get(url)
        time.sleep(5)  # Wait for content to load

        text = driver.find_element(By.TAG_NAME, 'body').text
        driver.quit()
        return text

    except Exception as e:
        return f"Error scraping LinkedIn: {e}"
    
# Extract text from image using OCR (Tesseract OCR for png, .jpg, .jpeg)
import pytesseract
from PIL import Image
from io import BytesIO

def extract_text_from_image(file) -> str:
    try:
        image = Image.open(BytesIO(file.file.read()))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from image: {e}")
    
####################################### Extractors #######################################


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
    






############################################################################################
####################################### API endpoint #######################################
############################################################################################


# API Endpoint for Indication Matching
@app.get("/match/indication")
def match_indication(input: str = Query(..., description="Free-text indication to match")):
    # Use RapidFuzz to find top 3 matches
    choices = df["Dx"].tolist()
    results = process.extract(input, choices, scorer=fuzz.token_sort_ratio, limit=3)

    matches = []
    for match_text, score, idx in results:
        snomed_term = df.iloc[idx]["Dx"]
        matches.append({
            "input_phrase": input,
            "matched_text": match_text,
            "snomed_term": snomed_term,
            "similarity_score": score
        })

    return {"results": matches}

# API endpoint@app.post("/parse")
@app.post("/parse")
async def parse_resume(file: UploadFile = File(...), snomed_matches: int = 3):
    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.content_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
        "application/msword"
    ]:
        text = extract_text_from_docx(file)
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    try:
        # Preprocess
        text = preprocess_text(text)

        # Resume parsing
        try:
            extracted_data = extract_data_with_gpt(text)
            overall_score = calculate_overall_confidence(extracted_data)
            extracted_data["overall_confidence"] = overall_score
        except Exception as e:
            return JSONResponse(content={"error": f"GPT extraction failed: {e}"}, status_code=500)

        # SNOMED embedding match
        try:
            input_emb = get_embedding(text)
            df_copy = df.copy()
            df_copy["similarity"] = df_copy["embedding"].apply(lambda e: cosine_similarity(e, input_emb))
            top_matches = df_copy.sort_values(by="similarity", ascending=False).head(snomed_matches)

            snomed_results = []
            for _, row in top_matches.iterrows():
                snomed_results.append({
                    "matched_text": row["Dx"],
                    "snomed_code": row["SNOMED CT Code"],
                    "similarity_score": round(row["similarity"], 4),
                    "abbreviation": row.get("Abbreviation", None)
                })
        except Exception as e:
            return JSONResponse(content={"error": f"SNOMED matching failed: {e}"}, status_code=500)

        return {
            "extracted_fields": extracted_data,
            "snomed_matches": snomed_results
        }

    except Exception as e:
        return JSONResponse(content={"error": f"General failure: {e}"}, status_code=500)
    


@app.post("/parse/Linkedin_url")
async def parse_resume_from_url(url: str):
    try:
        text = extract_text_from_linkedin(url)
        text = preprocess_text(text)
        extracted_data = extract_data_with_gpt(text)
        extracted_data["overall_confidence"] = calculate_overall_confidence(extracted_data)
        return extracted_data
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

from typing import List
from fastapi import UploadFile, File
from PIL import Image
import pytesseract
import io

@app.post("/parse/images")
async def parse_multiple_images(files: List[UploadFile] = File(...)):
    extracted_texts = []

    for file in files:
        if file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(await file.read()))
            text = pytesseract.image_to_string(image)
            extracted_texts.append(text)

    combined_text = preprocess_text("\n\n".join(extracted_texts))
    extracted_data = extract_data_with_gpt(combined_text)
    overall_score = calculate_overall_confidence(extracted_data)
    extracted_data["overall_confidence"] = overall_score
    return extracted_data

    


# Optional root route
@app.get("/")
def root():
    return {"message": "Resume Parser is running. Visit /docs to test."}   



