import os 
import pandas as pd 
from src.exception import CustomException
from src.logging import logging
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.component.data_transformation import DataTransformation 

## Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv') 
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

## Create a Class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')
        try:
            df = pd.read_csv('notebook/data/Data.csv')
            logging.info('Dataset read as Pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw Data saved')

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            logging.info('Train Test split done.')

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            logging.info('Train Data Saved')

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True)
            logging.info('Test Data Saved and Data Ingestion is completed.')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception Occurred at Data Ingestion stage')
            raise CustomException(e, sys)
        

    
