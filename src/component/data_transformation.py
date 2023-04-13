import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils import save_object

from src.exception import CustomException
from src.logging import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")
            
            cat_col = ['cut', 'color', 'clarity']
            num_col = ['carat', 'depth', 'table', 'x', 'y', 'z']

            #Define custom ranking of each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Inititated')

            #Pipeline Estsblishement
            # Numerical Pipeline
            num_pipeline = Pipeline(
            steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
            ])

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoding', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_col),
                ('cat_pipeline', cat_pipeline, cat_col)
            ])

            return preprocessor
        
            logging.info("Pipeline Execution Ends")

        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Data Transformation Initiated')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Data Reading completed')
            logging.info(f'Train Dataframe head : \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe head : \n {test_df.head().to_string()}')

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_col = 'price'
            drop_col = [target_col, 'id']

            input_feature_train_df = train_df.drop(columns=drop_col)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns= drop_col)
            target_feature_test_df = test_df[target_col]

            #Transforming using preprocessor object

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing dataset')


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            logging.info("Preprocessor pickle file Saved")
            
            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception Occurred in the initiate_data_transformation")
            raise CustomException(e, sys)