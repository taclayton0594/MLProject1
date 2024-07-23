import os
import sys
from dataclasses import dataclass
# from catboost import CatBoostRegressor
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
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_mdls

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting data into train and test.")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # Create dictionary of models to try
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K Nearest Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Adaboost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_mdls(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            # Best model performance
            best_model_perf = max(sorted(model_report.values()))

            # Best model name from dictionary
            best_model_name = list(model_report.keys())[
                            list(model_report.values()).index(best_model_perf)
                            ]
            
            # Best model
            best_model = models[best_model_name]

            if best_model_perf < 0.6:
                raise CustomException("All models do not meet minimum performance.")            

            logging.info(f"Best model has been found that meets minimum performance.")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
