# 🧠 Resume Parser PoC (GPT + LandingAI)

A FastAPI-powered proof-of-concept (PoC) application that parses resume data from PDF, DOCX, images, or LinkedIn URLs using a hybrid of OpenAI's GPT models and LandingAI's OCR API.

---

## 🔧 Features

### ✅ Input Support

1. **PDF / DOCX Resumes**

   * Parsed with PyMuPDF and `python-docx`
   * Cleaned and structured using OpenAI GPT
   * Returns key fields with confidence scores

2. **LinkedIn URLs**

   * Scraped using headless Selenium
   * Parsed using GPT
   * ⚠️ Subject to scraping limitations (bot protection, dynamic content)
   * Known issue: LLM may hallucinate/fill missing data (e.g., adding nonexistent roles)

3. **Image Resumes (PNG / JPG / JPEG)**

   * Uploaded as single or multiple images
   * Parsed using [LandingAI’s Resume Parser API](https://landing.ai)
   * Ideal for scanned or photographed resumes

---

## 📦 API Endpoints

| Method | Endpoint              | Description                                   |
| ------ | --------------------- | --------------------------------------------- |
| POST   | `/parse`              | Upload and parse a PDF or DOCX file           |
| POST   | `/parse/Linkedin_url` | Submit a LinkedIn profile URL for scraping    |
| POST   | `/parse/images`       | Upload 1+ image files to parse with LandingAI |
| GET    | `/`                   | Health check for server                       |

---

## 🧪 Example Usage

Start your local FastAPI server with:

```bash
uvicorn main2:app --reload
```

Visit Swagger UI to test:
📍 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ⚙️ Environment Setup

Create a `.env` file in the root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_key
VISION_AGENT_API_KEY=your_landingai_key
```

Install dependencies and run the server:

```bash
pip install -r requirements.txt
uvicorn main2:app --reload
```

---

## 🧠 Technologies Used

* **FastAPI** – lightweight API framework
* **OpenAI GPT** – resume field extraction from text
* **LandingAI** – resume OCR via Vision Agent

---

## ⚠️ Known Issues

* LinkedIn scraping may result in incomplete data or fail due to privacy restrictions
* LLM hallucination possible when input is unstructured or minimal
* LandingAI API requires valid key and resumes must be image files only

---

## 🛍️ Future Improvements

* Handle multilingual or low-data resumes
* Add fallback test pages for LinkedIn if scraping fails
* Explore Claude or other LLMs for richer comparisons
* Add user interface for manual corrections on outputs

---

## ✍️ Author

Built by **Hitomi Hoshino**
💡 Focused on resume intelligence, OCR evaluation, and hybrid AI pipelines.
