import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Validate if the dataset has all required columns
        """
        required_columns = [
            'Machine_ID',
            'Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
            'Air_System_Pressure(bar)', 'Coolant_Temperature',
            'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
            'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)',
            'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)', 'Downtime'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def initiate_data_ingestion(self, file_path: str = None) -> str:
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            if file_path:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                df = pd.read_csv(file_path)
            else:
                default_path = os.path.join('artifacts', 'data.csv')
                if not os.path.exists(default_path):
                    raise FileNotFoundError(f"Default data file not found: {default_path}")
                df = pd.read_csv(default_path)

            logging.info('Read the dataset as dataframe')

            # Initial preprocessing
            # Drop non-essential columns if they exist
            columns_to_drop = ['Assembly_Line_No', 'Date']
            existing_columns = [col for col in columns_to_drop if col in df.columns]
            if existing_columns:
                df = df.drop(existing_columns, axis=1)

            # Validate dataset columns after preprocessing
            self.validate_dataset(df)

            # Rename machines
            df['Machine_ID'] = df['Machine_ID'].replace({
                'Makino-L1-Unit1-2013': 'M1',
                'Makino-L2-Unit1-2015': 'M2',
                'Makino-L3-Unit1-2015': 'M3'
            })
            
            # Drop any missing values
            df = df.dropna()
            
            logging.info('Initial preprocessing completed')

            # Create directory for saving files
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data is saved in artifacts folder")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            logging.error("Exception occurred during Data Ingestion")
            raise CustomException(e, sys)