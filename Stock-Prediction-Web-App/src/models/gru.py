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

class GRU(keras.Model):
    def __init__(self,num_days,num_hid,time_steps,kernel_size):
        logging.info('GRU model initialising...')
        super().__init__()
        self.num_days = num_days
        self.num_hid = num_hid
        self.input_layer = layers.Input((time_steps, self.num_hid))
        self.time2vec = Time2Vec(kernel_size = kernel_size)
        self.gru_input = layers.Input((time_steps, self.num_hid + kernel_size))

        self.gru = keras.Sequential(
            [
                self.gru_input,
                layers.GRU(256),
                layers.Dense(units=num_days*num_hid, activation='linear')
            ]
        )

    def call(self, inputs):
        x1 = inputs
        x2 = self.time2vec(x1)
        x = tf.concat([x1,x2], axis=2)
        x = self.gru(x)

        shape = tf.shape(x)
        first_dim = tf.reduce_prod(shape) // (self.num_days*self.num_hid)
        x = tf.reshape(x,(first_dim,self.num_days,self.num_hid))
        return x
    
class GRU_Model():
    def __init__(self, num_days) -> None:
        try:
            self.num_days = num_days
            self.model = GRU(num_days=self.num_days,num_hid=5,time_steps=10,kernel_size=3)
        except Exception as e:
            logging.info('Failed while initialising GRU model...')
            raise CustomException(e,sys)
        
    def train(self, train_x, train_y):
        logging.info('GRU model training started...')
        try :
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
            self.model.compile(optimizer=opt, loss='mse')

            history = self.model.fit(train_x,train_y,epochs =5)
            logging.info(train_x[:5])
        except Exception as e:
            logging.info('Failed while training GRU Model...')
            raise CustomException(e,sys)

    def test(self, test_x, test_y):
        logging.info('Testing GRU model...')
        return self.model.evaluate(test_x,test_y)

    def Predict(self, test_x):
        logging.info('GRU model predicting for previous data...')
        return self.model.predict(test_x)
    
    def prediction(self,input_sequences, targets, prediction_data):
        logging.info('Stock Price prediction started using GRU Model...')
        try :
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
            self.model.compile(optimizer=opt, loss='mse')
            history = self.model.fit(input_sequences, targets,epochs =3)
            prediction = self.model.predict(prediction_data)
            return prediction
        except Exception as e:
            logging.info('GRU Model failed while prediction...')
            raise CustomException(e,sys)
        