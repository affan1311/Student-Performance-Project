from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

import os
import sys

from src.exception import CustomException
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)  # Train model
            y_train_pred = model.predict(X_train)  # Predict on training data
            y_test_pred = model.predict(X_test)  # Predict on test data
            train_model_score=r2_score(y_train,y_train_pred)  # R2 score on training data
            test_model_score=r2_score(y_test,y_test_pred)  # R2 score on test data
            report[list(models.keys())[i]] = test_model_score  # Store test R2 score in report
            
        return report

    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {e}")
        raise CustomException(e, sys)