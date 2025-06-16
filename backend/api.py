from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io


import os
import gdown

model_dir = "model"
model_path = os.path.join(model_dir, "gender_model.h5")

# Google Drive file download
if not os.path.exists(model_path):
    print("🔄 Downloading model from Google Drive...")
    os.makedirs(model_dir, exist_ok=True)
    # This is the Google Drive shareable link ID (not the full URL)
    file_id = "1TwCwEVjHkG3lkyif4mywdTmXJ10nFhe4"
    
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the trained model
print("✅ Loading model...")
model = load_model(model_path)
class_labels = ["female", "male"]

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(predictions))

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
