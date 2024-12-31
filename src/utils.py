import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

#from src.exception import CustomException


from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            para = param.get(model_name, {})
            
            try:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(x_train, y_train)
                
                best_model = gs.best_estimator_
                y_train_pred = best_model.predict(x_train)
                y_test_pred = best_model.predict(x_test)
                
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                # Only add successful model evaluations to report
                if test_model_score is not None:
                    report[model_name] = test_model_score
                    logging.info(f"Model {model_name} R2 score: {test_model_score}")
                
            except Exception as e:
                logging.warning(f"Failed to evaluate model {model_name}: {e}")
                continue
                
        if not report:
            raise CustomException("No models were successfully evaluated")
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)    