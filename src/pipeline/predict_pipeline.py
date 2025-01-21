import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def validate_input(self, features: Dict[str, Any]) -> None:
        """
        Validate input features
        """
        required_features = {
            'Machine_ID': str,
            'Hydraulic_Pressure': float,
            'Coolant_Pressure': float,
            'Air_System_Pressure': float,
            'Coolant_Temperature': float,
            'Hydraulic_Oil_Temperature': float,
            'Spindle_Bearing_Temperature': float,
            'Spindle_Vibration': float,
            'Tool_Vibration': float,
            'Spindle_Speed': float,
            'Voltage': float,
            'Torque': float,
            'Cutting': float
        }

        # Check for missing features
        missing_features = [feat for feat in required_features if feat not in features]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Validate data types and ranges
        for feat, feat_type in required_features.items():
            try:
                features[feat] = feat_type(features[feat])
                if feat_type == float and (features[feat] < 0 or features[feat] > 1000000):
                    raise ValueError(f"Value for {feat} is out of reasonable range")
            except (ValueError, TypeError):
                raise ValueError(f"Invalid type for {feat}. Expected {feat_type.__name__}")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions using the trained model
        """
        try:
            # Validate input features
            self.validate_input(features)
            
            # Load preprocessor and model
            if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError("Model or preprocessor files not found. Please train the model first.")

            preprocessor_dict = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])

            # Apply preprocessing
            # Label encoding
            for column, le in preprocessor_dict['label_encoders'].items():
                if column in features_df.columns:
                    try:
                        features_df[column] = le.transform(features_df[column])
                    except ValueError:
                        # Handle unknown categories
                        logging.warning(f"Unknown category in {column}. Using -1 as default.")
                        features_df[column] = -1

            # Scale features
            scaled_features = preprocessor_dict['scaler'].transform(features_df)

            # Make prediction
            pred = model.predict(scaled_features)
            pred_proba = model.predict_proba(scaled_features)

            # Convert prediction using label encoder
            prediction = preprocessor_dict['label_encoders']['Downtime'].inverse_transform(pred)[0]
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