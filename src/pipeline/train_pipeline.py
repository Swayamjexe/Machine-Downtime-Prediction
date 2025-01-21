import os
import sys
import time
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from typing import Dict, Optional

class TrainingPipeline:
    def __init__(self):
        self.artifact_dir = "artifacts"
        os.makedirs(self.artifact_dir, exist_ok=True)

    def validate_training_status(self) -> bool:
        """
        Check if model training is already in progress
        """
        lock_file = os.path.join(self.artifact_dir, "training.lock")
        if os.path.exists(lock_file):
            # Check if the lock file is stale (older than 1 hour)
            if os.path.getmtime(lock_file) < time.time() - 3600:
                os.remove(lock_file)
                return True
            return False
        return True

    def create_training_lock(self):
        """
        Create a lock file to indicate training is in progress
        """
        lock_file = os.path.join(self.artifact_dir, "training.lock")
        with open(lock_file, 'w') as f:
            f.write(str(time.time()))

    def remove_training_lock(self):
        """
        Remove the training lock file
        """
        lock_file = os.path.join(self.artifact_dir, "training.lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)

    def initiate_training(self, file_path: Optional[str] = None) -> Dict:
        """
        Execute the complete training pipeline
        """
        try:
            # Check if training is already in progress
            if not self.validate_training_status():
                raise CustomException("Training is already in progress. Please wait.", sys)

            # Create training lock
            self.create_training_lock()

            # Initialize components
            data_ingestion = DataIngestion()
            data_transformation = DataTransformation()
            model_trainer = ModelTrainer()

            try:
                # Data ingestion
                raw_data_path = data_ingestion.initiate_data_ingestion(file_path)
                logging.info("Data ingestion completed")

                # Data transformation
                train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(raw_data_path)
                logging.info("Data transformation completed")

                # Model training
                metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)
                logging.info("Model training completed")

                return {
                    "status": "success",
                    "message": "Training completed successfully",
                    "metrics": metrics,
                    "artifacts": {
                        "preprocessor": preprocessor_path,
                        "model": model_trainer.model_trainer_config.trained_model_file_path
                    }
                }

            finally:
                # Always remove the training lock when done
                self.remove_training_lock()

        except Exception as e:
            # Make sure to remove the lock file in case of any error
            self.remove_training_lock()
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)