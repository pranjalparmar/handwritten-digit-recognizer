from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
import io
from PIL import Image
import os
import warnings

# Suppress warnings
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize FastAPI app
app = FastAPI(
    title="Handwritten Digit Recognition API",
    description="Upload an image of a handwritten digit (0-9) and get the predicted digit",
    version="1.0.0"
)

# Load the pre-trained model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model('handwritten_recognition.keras')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess the uploaded image for digit recognition
    """
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale if needed
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # Convert PIL to numpy array
        img = np.array(pil_image)
        
        # Resize to 28x28 if needed
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))
        
        # Normalize pixel values to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Invert if needed (MNIST-style: white digit on black background)
        # Check if we need to invert by looking at the mean pixel value
        if np.mean(img) > 0.5:  # Likely black digit on white background
            img = 1 - img
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.post("/predict/", 
          summary="Predict handwritten digit",
          description="Upload an image containing a handwritten digit (0-9) to get the predicted digit and confidence score")
async def predict_digit(file: UploadFile = File(...)):
    """
    Predict the digit in an uploaded image
    
    - **file**: Image file containing a handwritten digit
    
    Returns the predicted digit and confidence score
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get predicted digit and confidence
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        # Get all probabilities
        probabilities = {str(i): float(prediction[0][i]) for i in range(10)}
        
        return JSONResponse(content={
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 4),
            "all_probabilities": probabilities,
            "filename": file.filename
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/", 
         summary="API Information",
         description="Get basic information about the API")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Handwritten Digit Recognition API",
        "description": "Upload an image of a handwritten digit to get predictions",
        "endpoints": {
            "predict": "/predict/ (POST)",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health",
         summary="Health Check",
         description="Check if the API and model are working properly")
async def health_check():
    """
    Health check endpoint
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    
    return {
        "status": "healthy",
        "message": "API and model are working properly"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)