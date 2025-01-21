import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
from sklearn.preprocessing import LabelEncoder

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.machine_le = LabelEncoder()
        self.machine_le.fit(['M1', 'M2', 'M3', 'Makino-L1-Unit1-2013', 'Makino-L2-Unit1-2015', 'Makino-L3-Unit1-2015'])
        
        # Define the expected feature order
        self.expected_features = [
            'Machine_ID',
            'Hydraulic_Pressure(bar)',
            'Coolant_Pressure(bar)',
            'Air_System_Pressure(bar)',
            'Coolant_Temperature',
            'Hydraulic_Oil_Temperature(?C)',
            'Spindle_Bearing_Temperature(?C)',
            'Spindle_Vibration(?m)',
            'Tool_Vibration(?m)',
            'Spindle_Speed(RPM)',
            'Voltage(volts)',
            'Torque(Nm)',
            'Cutting(kN)'
        ]

    def validate_input(self, features: Dict[str, Any]) -> None:
        required_features = set(self.expected_features)
        features = {k: v for k, v in features.items() if k not in ['Date', 'Assembly_Line_No']}
        
        if not required_features.issubset(features.keys()):
            missing = required_features - set(features.keys())
            raise ValueError(f"Missing required features: {list(missing)}")

        for feat, value in features.items():
            if feat != 'Machine_ID':
                try:
                    value = float(value)
                    if value < 0 or value > 1000000:
                        raise ValueError(f"Value for {feat} is out of reasonable range")
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid numeric value for {feat}")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        try:
            features = {k: v for k, v in features.items() if k not in ['Date', 'Assembly_Line_No']}
            self.validate_input(features)
            
            if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError("Model or preprocessor files not found. Please train the model first.")

            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            
            # Create DataFrame with features in the correct order
            features_df = pd.DataFrame([{feature: features[feature] for feature in self.expected_features}])
            
            # Apply label encoding to Machine_ID
            features_df['Machine_ID'] = self.machine_le.transform(features_df['Machine_ID'])

            # Scale features using the loaded preprocessor
            scaled_features = preprocessor.transform(features_df)

            # Make prediction
            pred = model.predict(scaled_features)
            pred_proba = model.predict_proba(scaled_features)

            prediction = "Yes" if pred[0] == 1 else "No"
            confidence = float(max(pred_proba[0]))

            return {
                "Downtime": prediction,
                "Confidence": confidence,
                "Probability_No": float(pred_proba[0][0]),
                "Probability_Yes": float(pred_proba[0][1])
            }

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)