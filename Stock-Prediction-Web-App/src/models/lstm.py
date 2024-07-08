import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers,models
import numpy as np
import sys

from src.logger import logging
from src.exception import CustomException

import warnings
warnings.filterwarnings('ignore')

from src.models.time2vec import Time2Vec

class LSTM(keras.Model):
    def __init__(self,num_days,num_hid,time_steps,kernel_size):
        logging.info('LSTM model initialising...')
        super().__init__()
        self.num_days = num_days
        self.num_hid = num_hid
        self.input_layer = layers.Input((time_steps, self.num_hid))
        self.time2vec = Time2Vec(kernel_size = kernel_size)
        self.lstm_input = layers.Input((time_steps, self.num_hid + kernel_size))

        self.lstm = keras.Sequential(
            [
                self.lstm_input,
                layers.LSTM(256),
                layers.Dense(units=num_days*num_hid, activation='linear')
            ]
        )

    def call(self, inputs):
        x1 = inputs
        x2 = self.time2vec(x1)
        x = tf.concat([x1,x2], axis=2)
        x = self.lstm(x)

        shape = tf.shape(x)
        first_dim = tf.reduce_prod(shape) // (self.num_days*self.num_hid)
        x = tf.reshape(x,(first_dim,self.num_days,self.num_hid))
        return x
    
class LSTM_Model():
    def __init__(self, num_days) -> None:
        try:
            self.num_days = num_days
            self.model = LSTM(num_days=self.num_days,num_hid=5,time_steps=10,kernel_size=3)
        except Exception as e:
            logging.info('Failed while initialising LSTM model...')
            raise CustomException(e,sys)
        
    def train(self, train_x, train_y):
        logging.info('LSTM model training started...')
        try :
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
            self.model.compile(optimizer=opt, loss='mse')

            history = self.model.fit(train_x,train_y,epochs =4)
        except Exception as e:
            logging.info('Failed while training LSTM Model...')
            raise CustomException(e,sys)

    def test(self, test_x, test_y):
        logging.info('Testing LSTM model...')
        return self.model.evaluate(test_x,test_y)
    
    def Predict(self, test_x):
        return self.model.predict(test_x)

    def prediction(self,input_sequences, targets, prediction_data):
        logging.info('Stock Price prediction started using LSTM Model...')
        try :
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
            self.model.compile(optimizer=opt, loss='mse')
            history = self.model.fit(input_sequences, targets,epochs =3)
            prediction = self.model.predict(prediction_data)
            return prediction
        except Exception as e:
            logging.info('LSTM Model failed while prediction...')
            raise CustomException(e,sys)
        