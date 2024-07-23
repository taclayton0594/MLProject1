import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function performs various transformation for numerical and categorical data.
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")), # imputer handles missing values
                    ("scaler",StandardScaler())
                ]
            )

            categorial_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_columns),
                    ("categorial_pipeline",categorial_pipeline,categorical_columns)
                ]
            )

            logging.info("Numerical column standard scaling pipeline created.")
            logging.info("Categorical column standard scaling pipeline created.")

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data has been imported.")

            logging.info("Obtaining preprocessing object...")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            inputs_train_df = train_df.drop(columns=[target_column_name],axis=1)
            inputs_test_df = test_df.drop(columns=[target_column_name],axis=1)

            output_train_df = train_df[target_column_name]
            output_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on train and test datasets.")

            inputs_train_arr = preprocessing_obj.fit_transform(inputs_train_df)
            inputs_test_arr = preprocessing_obj.transform(inputs_test_df)

            train_arr = np.c_[inputs_train_arr, np.array(output_train_df)] # 'c_' is shorthand notation for concatenating arrays along second axis
            test_arr = np.c_[inputs_test_arr, np.array(output_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

