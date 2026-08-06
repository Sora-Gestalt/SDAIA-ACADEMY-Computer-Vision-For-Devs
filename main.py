from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Damaged Apples")

# Load model weights once on startup
model = YOLO("best.pt")

@app.get("/")
def read_root():
    return {"status": "Damaged Apples Detection API is live"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Run YOLO model prediction
    results = model(image)
    res = results[0]
    
    predictions = []
    boxes = res.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = res.names[class_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
        
        predictions.append({
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": [round(coord, 2) for coord in bbox]
        })
        
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)