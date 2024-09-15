from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
        return ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]
    elif provider == "OpenRouter":
        return ["openai/chatgpt-4o-latest", "openai/gpt-4o-mini-2024-07-18","mistralai/pixtral-12b:free","meta-llama/llama-3.1-405b","google/gemini-pro-vision"]
    else:
        raise HTTPException(status_code=400, detail="Invalid provider")

@app.post("/ocr")
async def process_ocr(
    files: List[UploadFile] = File(...), 
    provider: str = Form(...), 
    model: str = Form(...),
    output_format: str = Form(...)
):
    try:
        logger.info(f"Received OCR request: provider={provider}, model={model}, output_format={output_format}")
        
        all_json_data = []
        
        for file in files:
            file_content = await file.read()
            
            if file.filename.lower().endswith('.pdf'):
                logger.info(f"Converting PDF to images: {file.filename}")
                images = convert_from_bytes(file_content)
                for i, image in enumerate(images):
                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format='JPEG')
                    image_bytes = image_bytes.getvalue()
                    
                    logger.info(f"Processing PDF page {i+1} with {provider}")
                    if provider == "OpenAI":
                        json_data = process_image_openai(image_bytes, model)
                    elif provider == "OpenRouter":
                        json_data = process_image_openrouter(image_bytes, model)
                    else:
                        raise ValueError(f"Unsupported provider: {provider}")
                    
                    all_json_data.append({"filename": file.filename, "page": i+1, "content": json_data})
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

        if output_format == "json":
            output = json.dumps(merged_json_data, indent=2, ensure_ascii=False)
            media_type = "application/json"
        else:
            output = json.dumps(merged_json_data, indent=2, ensure_ascii=False)
            media_type = "text/plain"

        logger.info("OCR processing completed successfully")
        
        # Create a Response object with formatted JSON
        response = JSONResponse(content=json.loads(output), media_type=media_type)
        
        # Set content disposition to force download
        response.headers["Content-Disposition"] = f"attachment; filename=ocr_output.{output_format}"
        
        return response

    except Exception as e:
        logger.error(f"Error in process_ocr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)
