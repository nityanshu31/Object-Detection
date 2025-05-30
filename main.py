from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import shutil
import os

app = FastAPI()

# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client("1aurent/cogvlm_captionner")

@app.post("/generate-caption")
async def generate_caption(file: UploadFile = File(...)):
    file_location = f"./temp_{file.filename}"
    
    # Save the uploaded image
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Use handle_file on the saved local path
        result = client.predict(
            image=handle_file(file_location),
            query="Provide a factual description of this image in up to two paragraphs. Include details on objects, background, scenery, interactions, gestures, poses, and any visible text content. Specify the number of repeated objects. Describe the dominant colors, color contrasts, textures, and materials. Mention the composition, including the arrangement of elements and focus points. Note the camera angle or perspective, and provide any identifiable contextual information. Include details on the style, lighting, and shadows. Avoid subjective interpretations or speculation.",
            api_name="/generate_caption"
        )
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Optional: cleanup temp file
        os.remove(file_location)

    return {"caption": result}
