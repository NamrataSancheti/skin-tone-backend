from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ml_model import process_selected_pixels

app = FastAPI(title="Skin Tone Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Skin Tone Detection API is running"}

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    region_type: str = "face"
):
    result = process_selected_pixels(image, mask, region_type)
    return result
