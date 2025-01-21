import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object successfully saved to {file_path}")
            
    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models using GridSearchCV and return performance metrics
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]
            
            logging.info(f"Starting GridSearchCV for {model_name}")
            
            gs = GridSearchCV(
                model,
                para,
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            
            gs.fit(X_train, y_train)
            
            # Set best parameters found by GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Log the performance metrics
            logging.info(f"{model_name} Train Score: {train_model_score}")
            logging.info(f"{model_name} Test Score: {test_model_score}")
            logging.info(f"{model_name} Best Parameters: {gs.best_params_}")
            
            report[model_name] = test_model_score
            
        return report

    except Exception as e:
        logging.error("Error occurred while evaluating models")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a file using pickle
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            logging.info(f"Object successfully loaded from {file_path}")
            return obj
            
    except Exception as e:
        logging.error(f"Error occurred while loading object from {file_path}")
        raise CustomException(e, sys)

def get_model_metrics(y_true, y_pred, is_classification=True):
    """
    Calculate various model performance metrics
    """
    try:
        metrics = {}
        
        # Always calculate R2 score
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # If it's a classification problem, calculate classification metrics
        if is_classification:
            metrics.update({
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            })
            
        return metrics
        
    except Exception as e:
        logging.error("Error occurred while calculating metrics")
        raise CustomException(e, sys)