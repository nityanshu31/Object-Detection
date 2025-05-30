from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
import shutil
import os

app = FastAPI()

# Initialize Gradio client once
gr_client = Client("GanymedeNil/Qwen2-VL-7B")

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to disk
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Send to Gradio Space
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
