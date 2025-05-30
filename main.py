# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client, handle_file

app = FastAPI()

client = Client("1aurent/cogvlm_captionner")

class ImageURLRequest(BaseModel):
    image_url: str

@app.post("/generate-caption")
async def generate_caption(data: ImageURLRequest):
    try:
        result = client.predict(
            image=handle_file(data.image_url),
            query=(
                "Provide a factual description of this image in up to two paragraphs. "
                "Include details on objects, background, scenery, interactions, gestures, poses, and any visible text content. "
                "Specify the number of repeated objects. Describe the dominant colors, color contrasts, textures, and materials. "
                "Mention the composition, including the arrangement of elements and focus points. "
                "Note the camera angle or perspective, and provide any identifiable contextual information. "
                "Include details on the style, lighting, and shadows. Avoid subjective interpretations or speculation."
            ),
            api_name="/generate_caption"
        )
        return {"caption": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
