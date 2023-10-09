import os, sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split data into training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1], 
                test_array[:,:-1], 
                test_array[:,-1]
            )

            models = {
                "Random_Forest": RandomForestRegressor(),
                "Decision_Tree": DecisionTreeRegressor(),
                "Gradient_Boosting": GradientBoostingRegressor(),
                "Linear_Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
                "AdaBoost": AdaBoostRegressor(),

            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # get best model
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"Best model score: {best_model_score}")

            # best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f"Best model name: {best_model}")
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            save_obj(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = best_model
            )
        
            predicted = best_model.predict(X_test)
            r_square = r2_score(y_test, predicted)
            logging.info(f"R2 Score of best model: {r_square}")

            return r_square

        
        except Exception as e:
            raise CustomException(e,sys)

