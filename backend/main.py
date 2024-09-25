from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from dotenv import load_dotenv
import base64
import json
import requests
from openai import OpenAI
from pdf2image import convert_from_bytes
import io
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in the environment variables.")
    raise ValueError("OPENAI_API_KEY is not set")

if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY is not set in the environment variables.")
    raise ValueError("OPENROUTER_API_KEY is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def process_image_openai(image_bytes, model):
    try:
        image_url = f"data:image/jpeg;base64,{encode_image(image_bytes)}"
        
        response = openai_client.chat.completions.create(
            model=model, 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Return JSON document with data extracted from this image. Only return JSON, not other text."},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ],
                }
            ],
            max_tokens=500,
        )

        json_string = response.choices[0].message.content
        json_string = json_string.replace("```json\n", "").replace("\n```", "")
        json_data = json.loads(json_string)

        return json_data
    except Exception as e:
        logger.error(f"Error in process_image_openai: {str(e)}")
        raise

def process_image_openrouter(image_bytes, model):
    try:
        image_url = f"data:image/jpeg;base64,{encode_image(image_bytes)}"
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Return JSON document with data extracted from this image. Only return JSON, not other text."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        json_string = response.json()['choices'][0]['message']['content']
        json_string = json_string.replace("```json\n", "").replace("\n```", "")
        json_data = json.loads(json_string)

        return json_data
    except Exception as e:
        logger.error(f"Error in process_image_openrouter: {str(e)}")
        raise

@app.get("/models/{provider}")
async def get_models(provider: str):
    if provider == "OpenAI":
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    elif provider == "OpenRouter":
        return ["openai/chatgpt-4o-2024-08-06", "openai/gpt-4o-mini", "mistralai/pixtral-12b:free", "anthropic/claude-3.5-sonnet", "google/gemini-flash-8b-1.5-exp", "google/gemini-pro-1.5-exp", "google/gemini-pro-vision"]
    else:
        raise HTTPException(status_code=400, detail="Invalid provider")

def custom_json_format(json_str):
    # Add a newline after each closing curly brace, except the last one
    formatted = re.sub(r'}(?!}*$)', '}\n', json_str)
    return formatted

@app.post("/ocr")
async def process_ocr(
    files: List[UploadFile] = File(...), 
    provider: str = Form(...), 
    model: str = Form(...),
    output_format: str = Form(...)
):
    try:
        logger.info(f"Received OCR request: provider={provider}, model={model}, output_format={output_format}")
        
        def merge_json(existing, new):
            if isinstance(existing, dict) and isinstance(new, dict):
                for key, value in new.items():
                    if key in existing:
                        existing[key] = merge_json(existing[key], value)
                    else:
                        existing[key] = value
                return existing
            elif isinstance(existing, list) and isinstance(new, list):
                return existing + new
            elif isinstance(existing, str) and isinstance(new, str):
                return existing.rstrip('.,!?') + ' ' + new.lstrip()
            else:
                return new

        all_json_data = []
        
        for file in files:
            file_content = await file.read()
            
            if file.filename.lower().endswith('.pdf'):
                logger.info(f"Converting PDF to images: {file.filename}")
                images = convert_from_bytes(file_content)
                pdf_json_data = {}
                for image in images:
                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format='JPEG')
                    image_bytes = image_bytes.getvalue()
                    
                    logger.info(f"Processing PDF page with {provider}")
                    if provider == "OpenAI":
                        page_json_data = process_image_openai(image_bytes, model)
                    elif provider == "OpenRouter":
                        page_json_data = process_image_openrouter(image_bytes, model)
                    else:
                        raise ValueError(f"Unsupported provider: {provider}")
                    
                    # Merge page_json_data into pdf_json_data
                    pdf_json_data = merge_json(pdf_json_data, page_json_data)
                
                all_json_data.append({"filename": file.filename, "content": pdf_json_data})
            else:
                image_bytes = file_content
                logger.info(f"Processing image with {provider}: {file.filename}")
                if provider == "OpenAI":
                    json_data = process_image_openai(image_bytes, model)
                elif provider == "OpenRouter":
                    json_data = process_image_openrouter(image_bytes, model)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
                
                all_json_data.append({"filename": file.filename, "content": json_data})

        # Merge all JSON data
        merged_json_data = {
            "files": all_json_data
        }

        # Convert to JSON string without any formatting
        json_str = json.dumps(merged_json_data, ensure_ascii=False, separators=(',', ':'))

        # Apply custom formatting
        formatted_json = custom_json_format(json_str)

        # Create a bytes IO object
        json_bytes = io.BytesIO(formatted_json.encode('utf-8'))

        # Create a response with the formatted JSON
        response = Response(content=json_bytes.getvalue(), media_type="application/json")
        
        # Set headers to force download
        response.headers["Content-Disposition"] = f"attachment; filename=ocr_output.json"
        response.headers["Content-Type"] = "application/json; charset=utf-8"

        logger.info("OCR processing completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in process_ocr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)
