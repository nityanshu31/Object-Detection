from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
import shutil
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

# Initialize Gradio client with Hugging Face token
gr_client = Client("GanymedeNil/Qwen2-VL-7B", hf_token=HF_TOKEN)

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to disk
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call the Hugging Face model
        result = gr_client.predict(
            image=handle_file(file_location),
            text_input=None,
            model_id="Qwen/Qwen2-VL-7B-Instruct",
            api_name="/run_example"
        )

        # Clean up local file
        os.remove(file_location)

        return JSONResponse(content={"result": result})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
