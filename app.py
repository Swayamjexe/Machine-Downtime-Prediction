import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict
import pandas as pd
import io
import shutil
from datetime import datetime
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import logging

app = FastAPI(
    title="Manufacturing Predictive Analysis API",
    description="API for predicting machine downtime in manufacturing operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    Machine_ID: str = Field(..., description="Machine identifier")
    Hydraulic_Pressure: float = Field(..., alias='Hydraulic_Pressure(bar)', ge=0, description="Hydraulic pressure reading in bar")
    Coolant_Pressure: float = Field(..., alias='Coolant_Pressure(bar)', ge=0, description="Coolant pressure reading in bar")
    Air_System_Pressure: float = Field(..., alias='Air_System_Pressure(bar)', ge=0, description="Air system pressure reading in bar")
    Coolant_Temperature: float = Field(..., ge=0, description="Coolant temperature reading")
    Hydraulic_Oil_Temperature: float = Field(..., alias='Hydraulic_Oil_Temperature(?C)', ge=0, description="Hydraulic oil temperature in °C")
    Spindle_Bearing_Temperature: float = Field(..., alias='Spindle_Bearing_Temperature(?C)', ge=0, description="Spindle bearing temperature in °C")
    Spindle_Vibration: float = Field(..., alias='Spindle_Vibration(?m)', ge=0, description="Spindle vibration reading in µm")
    Tool_Vibration: float = Field(..., alias='Tool_Vibration(?m)', ge=0, description="Tool vibration reading in µm")
    Spindle_Speed: float = Field(..., alias='Spindle_Speed(RPM)', ge=0, description="Spindle speed in RPM")
    Voltage: float = Field(..., alias='Voltage(volts)', ge=0, description="Voltage reading in volts")
    Torque: float = Field(..., alias='Torque(Nm)', ge=0, description="Torque reading in Nm")
    Cutting: float = Field(..., alias='Cutting(kN)', ge=0, description="Cutting force in kN")

    class Config:
        allow_population_by_field_name = True


    @validator('Machine_ID')
    def validate_machine_id(cls, v):
        valid_ids = ['M1', 'M2', 'M3', 'Makino-L1-Unit1-2013', 'Makino-L2-Unit1-2015', 'Makino-L3-Unit1-2015']
        if v not in valid_ids:
            raise ValueError(f'Machine_ID must be one of {valid_ids}')
        return v

FILE_SIZE_LIMIT = 10 * 1024 * 1024  # 10MB

def cleanup_old_files(directory: str, max_files: int = 5):
    """Clean up old files keeping only the most recent ones"""
    if not os.path.exists(directory):
        return
    
    files = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            files.append((path, os.path.getmtime(path)))
    
    # Sort files by modification time
    files.sort(key=lambda x: x[1], reverse=True)
    
    # Remove old files
    for path, _ in files[max_files:]:
        try:
            os.remove(path)
            logging.info(f"Cleaned up old file: {path}")
        except Exception as e:
            logging.error(f"Error cleaning up {path}: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a CSV file containing manufacturing data
    """
    try:
        # Validate file size
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        
        if size > FILE_SIZE_LIMIT:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")

        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        upload_path = os.path.join("uploads", filename)

        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save the file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read and validate file content
        try:
            df = pd.read_csv(upload_path)
            required_columns = ['Machine_ID', 'Downtime']
            if not all(col in df.columns for col in required_columns):
                os.remove(upload_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required columns. Required: {required_columns}"
                )
        except Exception as e:
            os.remove(upload_path)
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

        # Schedule cleanup task
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, "uploads")

        return JSONResponse(
            content={
                "status": "success",
                "message": "File uploaded successfully",
                "filename": filename,
                "shape": df.shape,
                "columns": list(df.columns)
            }
        )

    except Exception as e:
        logging.error(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(file: UploadFile = None):
    """
    Train the model on the uploaded dataset
    """
    try:
        pipeline = TrainingPipeline()
        
        if file:
            # If new file is uploaded for training
            file_path = os.path.join("uploads", file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            result = pipeline.initiate_training(file_path=file_path)
        else:
            # Use default dataset
            result = pipeline.initiate_training()

        return JSONResponse(
            content={
                "status": "success",
                "message": "Model training completed successfully",
                "metrics": result
            }
        )

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Make predictions using the trained model
    """
    try:
        pipeline = PredictionPipeline()
        features = data.dict()
        
        prediction = pipeline.predict(features)
        
        return JSONResponse(content=prediction)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check if the API is running and model files exist
    """
    model_exists = os.path.exists(os.path.join("artifacts", "model.pkl"))
    preprocessor_exists = os.path.exists(os.path.join("artifacts", "preprocessor.pkl"))
    
    return {
        "status": "healthy",
        "model_ready": model_exists and preprocessor_exists,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)