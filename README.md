# Manufacturing Predictive Analysis API

A FastAPI-based REST API for predicting machine downtime in manufacturing operations. This project uses machine learning to analyze manufacturing data and predict potential machine downtime, helping optimize maintenance schedules and reduce unexpected failures.

## Features

- Data upload endpoint for CSV files containing manufacturing metrics
- Model training endpoint with performance metrics feedback
- Real-time prediction endpoint for machine downtime
- Automated data validation and preprocessing
- Multiple machine learning models comparison (Random Forest, XGBoost, CatBoost, etc.)
- Model performance metrics including accuracy, F1-score, and cross-validation results

## Technologies Used

- Python 3.11.1
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- XGBoost
- CatBoost
- Uvicorn

## Project Structure

```
├── artifacts/             # Trained models and preprocessors
|   ├── data.csv
|   ├── model.pkl          # Saved after Training is completed
|   └── preprocessor.pkl   # Saved after Data Transformation step is completed   
├── src/
│   ├── components/       # Core components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/        # Pipeline modules
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py     # Custom exception handling
│   ├── logger.py       # Logging configuration
│   └── utils.py        # Utility functions
├── uploads/             # Uploaded datasets
├── app.py              # FastAPI application
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Swayamjexe/Machine-Downtime-Prediction.git
cd Machine-Downtime-Prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
OR
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Upload Data (`POST /upload`)

Upload a CSV file containing manufacturing data.

#### cURL
```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/data.csv"
```

#### Postman Instructions
1. Create a new POST request to `http://localhost:8000/upload`
2. Go to the "Body" tab
3. Select "form-data"
4. Add a key "file" of type "File"
5. Select your CSV file
6. Send the request

### 2. Train Model (`POST /train`)

Train the machine learning model on the uploaded dataset.

#### cURL
```bash
curl -X POST http://localhost:8000/train
```

#### Postman Instructions
1. Create a new POST request to `http://localhost:8000/train`
2. Send the request

### 3. Make Prediction (`POST /predict`)

Get downtime predictions for machine parameters.

#### cURL
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Date": "31-12-2021",
    "Machine_ID": "Makino-L1-Unit1-2013",
    "Assembly_Line_No": "Shopfloor-L1",
    "Hydraulic_Pressure(bar)": 71.04,
    "Coolant_Pressure(bar)": 6.933725,
    "Air_System_Pressure(bar)": 6.284965,
    "Coolant_Temperature": 25.6,
    "Hydraulic_Oil_Temperature(?C)": 46,
    "Spindle_Bearing_Temperature(?C)": 33.4,
    "Spindle_Vibration(?m)": 1.291,
    "Tool_Vibration(?m)": 26.492,
    "Spindle_Speed(RPM)": 25892,
    "Voltage(volts)": 335,
    "Torque(Nm)": 24.05533,
    "Cutting(kN)": 3.58
}'
```

#### Postman Instructions
1. Create a new POST request to `http://localhost:8000/predict`
2. Go to the "Body" tab
3. Select "raw" and "JSON"
4. Paste the JSON payload above
5. Send the request

## API Response Screenshots

### Upload Endpoint Response
![Image](https://github.com/user-attachments/assets/1a382409-422a-4f44-a89c-97b19088f90b)

### Train Endpoint Response
[Add screenshot showing successful training response in Postman]

### Predict Endpoint Response
[Add screenshot showing successful prediction response in Postman]

## Model Performance

The current model achieves:
- Training Accuracy: 100%
- Test Accuracy: 99.16%
- F1 Score: 0.9916
- Cross-validation Mean: 0.9937
- Cross-validation Standard Deviation: 0.0039

## Features Used for Prediction

The model uses the following features:
- Machine_ID
- Hydraulic Pressure
- Coolant Pressure
- Air System Pressure
- Coolant Temperature
- Hydraulic Oil Temperature
- Spindle Bearing Temperature
- Spindle Vibration
- Tool Vibration
- Spindle Speed
- Voltage
- Torque
- Cutting Force

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
