import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.logger import logging
from src.exception import CustomException

class DataProcessing():
    def __init__(self) -> None:
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        logging.info('Data Processing initialised.')

    def feature_selection(self,df):
        cols = ['Open','High','Low','Volume','Close']
        df=df[cols]
        logging.info('Relevant features of data collected.')
        return df
    
    def transform(self,df):
        self.data = self.scaler.fit_transform(df)
        logging.info('Data is scaled.')
        return self.data

    def get_sequential_data(self,num_days):
        logging.info('Generating sequential data.')
        sequence_length =10
        input_sequences=[]
        targets=[]
        for i in range(len(self.data) - sequence_length - num_days):
            input_sequences.append(self.data[i:i+sequence_length])
            targets.append(self.data[i+sequence_length:i+sequence_length+num_days])

        input_sequences = np.array(input_sequences)
        targets = np.array(targets)
        return input_sequences,targets

    def get_train_test_data(self,input_sequences, targets):
        logging.info('Generating train test data.')
        n = len(input_sequences)

        train_x = input_sequences[0:int(n*0.80)]
        train_y = targets[0:int(n*0.80)]
        test_x = input_sequences[int(n*0.80):]
        test_y = targets[int(n*0.80):]
        
        return train_x,train_y,test_x,test_y
    
    def get_prediction_data(self):
        logging.info('Generating data for prediction.')
        prediction_data = self.data[-10:]
        return np.array([prediction_data])