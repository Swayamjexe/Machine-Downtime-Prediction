import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            scaler = StandardScaler()
            return scaler
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data_path):
        try:
            # Read the data
            df = pd.read_csv(data_path)
            logging.info("Read data completed")

            # Initialize label encoder
            le = LabelEncoder()
            
            # Apply label encoding to categorical columns
            for column in df.columns:
                if df[column].dtype == type(object):
                    df[column] = le.fit_transform(df[column])
            
            logging.info("Label Encoding completed")

            # Separate features and target
            X = df.drop(['Downtime'], axis=1)
            y = df['Downtime']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logging.info("Data splitting completed")

            # Initialize and fit preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            
            # Transform the features
            X_train_scaled = preprocessing_obj.fit_transform(X_train)
            X_test_scaled = preprocessing_obj.transform(X_test)
            
            logging.info("Data transformation completed")

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")

            # Combine features and target
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)