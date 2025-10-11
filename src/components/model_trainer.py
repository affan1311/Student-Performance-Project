from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self): 
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting train and test data")
            X_train,y_train,X_test,y_test = (train_array[:,:-1],train_array[:,-1],
                                             test_array[:,:-1],test_array[:,-1])
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'Random Forest Regression': RandomForestRegressor(),
                'K-Neighbors Regression': KNeighborsRegressor(),
                'Decision Tree Regression': DecisionTreeRegressor(),
                'XGB Regression': XGBRegressor(),
                'CatBoosting Regression': CatBoostRegressor(verbose=False),
                'AdaBoost Regression': AdaBoostRegressor() 
            }
            
            params={
                'Decision Tree Regression': {
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },

                'Random Forest Regression': {
                    'n_estimators': [100, 200],
                    # 'max_features': ['sqrt', 'log2'],
                    'criterion': ['squared_error', 'absolute_error']
                },

                'K-Neighbors Regression': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },

                'Linear Regression': {},

                'Ridge Regression': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                },

                'Lasso Regression': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'selection': ['cyclic', 'random']
                },

                'XGB Regression': {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                },

                'CatBoosting Regression': {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'iterations': [100, 200, 300]
                },

                'AdaBoost Regression': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                }   

            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return best_model_name, r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)
