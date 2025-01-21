import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from typing import Dict, Any

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray, 
                      model: Any) -> Dict[str, float]:
        """
        Evaluate the model using various metrics
        """
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict[str, float]:
        """
        Initiate the model training process
        """
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define models with hyperparameters
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=8,
                    random_state=42
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                ),
                'CatBoost Classifier': CatBoostClassifier(
                    iterations=100,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=False
                ),
                'XGBoost': XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                )
            }

            # Train and evaluate models
            model_report = {}
            
            for name, model in models.items():
                try:
                    logging.info(f"Training {name}")
                    metrics = self.evaluate_model(X_train, y_train, X_test, y_test, model)
                    model_report[name] = metrics
                    
                    logging.info(f"{name} metrics:")
                    for metric, value in metrics.items():
                        logging.info(f"{metric}: {value:.4f}")
                        
                except Exception as e:
                    logging.error(f"Error training {name}: {str(e)}")

            # Find best model based on test accuracy
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_accuracy'])
            best_model = models[best_model_name]
            best_model_metrics = model_report[best_model_name]

            logging.info(f"Best Model Found: {best_model_name}")
            logging.info("Best Model Metrics:")
            for metric, value in best_model_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_metrics

        except Exception as e:
            logging.error("Exception occurred in Model Training")
            raise CustomException(e, sys)