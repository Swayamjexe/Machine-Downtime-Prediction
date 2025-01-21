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
    Date: Optional[str] = None
    Machine_ID: str
    Assembly_Line_No: Optional[str] = None
    Hydraulic_Pressure: Optional[float] = Field(None, alias='Hydraulic_Pressure(bar)')
    Coolant_Pressure: Optional[float] = Field(None, alias='Coolant_Pressure(bar)')
    Air_System_Pressure: Optional[float] = Field(None, alias='Air_System_Pressure(bar)')
    Coolant_Temperature: float
    Hydraulic_Oil_Temperature: Optional[float] = Field(None, alias='Hydraulic_Oil_Temperature(?C)')
    Spindle_Bearing_Temperature: Optional[float] = Field(None, alias='Spindle_Bearing_Temperature(?C)')
    Spindle_Vibration: Optional[float] = Field(None, alias='Spindle_Vibration(?m)')
    Tool_Vibration: Optional[float] = Field(None, alias='Tool_Vibration(?m)')
    Spindle_Speed: Optional[float] = Field(None, alias='Spindle_Speed(RPM)')
    Voltage: Optional[float] = Field(None, alias='Voltage(volts)')
    Torque: Optional[float] = Field(None, alias='Torque(Nm)')
    Cutting: Optional[float] = Field(None, alias='Cutting(kN)')

    class Config:
        allow_population_by_field_name = True

    @validator('Machine_ID')
    def validate_machine_id(cls, v):
        valid_ids = ['M1', 'M2', 'M3', 'Makino-L1-Unit1-2013', 'Makino-L2-Unit1-2015', 'Makino-L3-Unit1-2015']
        if v not in valid_ids:
            raise ValueError(f'Machine_ID must be one of {valid_ids}')
        return v

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        # Convert aliased field names to original names if they exist
        if d.get('Hydraulic_Pressure') is not None:
            d['Hydraulic_Pressure(bar)'] = d.pop('Hydraulic_Pressure')
        if d.get('Coolant_Pressure') is not None:
            d['Coolant_Pressure(bar)'] = d.pop('Coolant_Pressure')
        if d.get('Air_System_Pressure') is not None:
            d['Air_System_Pressure(bar)'] = d.pop('Air_System_Pressure')
        if d.get('Hydraulic_Oil_Temperature') is not None:
            d['Hydraulic_Oil_Temperature(?C)'] = d.pop('Hydraulic_Oil_Temperature')
        if d.get('Spindle_Bearing_Temperature') is not None:
            d['Spindle_Bearing_Temperature(?C)'] = d.pop('Spindle_Bearing_Temperature')
        if d.get('Spindle_Vibration') is not None:
            d['Spindle_Vibration(?m)'] = d.pop('Spindle_Vibration')
        if d.get('Tool_Vibration') is not None:
            d['Tool_Vibration(?m)'] = d.pop('Tool_Vibration')
        if d.get('Spindle_Speed') is not None:
            d['Spindle_Speed(RPM)'] = d.pop('Spindle_Speed')
        if d.get('Voltage') is not None:
            d['Voltage(volts)'] = d.pop('Voltage')
        if d.get('Torque') is not None:
            d['Torque(Nm)'] = d.pop('Torque')
        if d.get('Cutting') is not None:
            d['Cutting(kN)'] = d.pop('Cutting')
        return d

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