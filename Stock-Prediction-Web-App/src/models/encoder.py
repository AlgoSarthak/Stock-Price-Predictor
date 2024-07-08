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

class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads=2, embed_dim=8, feed_forward_dim=128, rate=0.05):
        super().__init__()

        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(feed_forward_dim, activation="relu"),
                                     layers.Dense(64,activation="relu"),
                                     layers.Dense(embed_dim),])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class T2VTransformer(keras.Model):
    def __init__(self,num_days, num_hid=5,time_steps=10,num_head=4,kernel_size=3,num_feed_forward=128,num_layers_enc=6, rate=0.05):

        super().__init__()
        self.num_hid = num_hid
        self.num_days = num_days
        self.numlayers_enc = num_layers_enc
        self.input_layer = layers.Input((time_steps, self.num_hid))
        self.time2vec = Time2Vec(kernel_size = kernel_size)
        self.enc_input = layers.Input((time_steps, self.num_hid + kernel_size))

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_heads=num_head,embed_dim= self.num_hid + kernel_size, feed_forward_dim=num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )
        self.GlobalAveragePooling1D = layers.GlobalAveragePooling1D(data_format='channels_last')
        self.dropout = layers.Dropout(rate)
        self.out1 = layers.Dense(units=128, activation='linear')
        self.out2 = layers.Dense(units=num_hid*self.num_days, activation='linear')

    def call(self, inputs):
        x1 = inputs
        x2 = self.time2vec(x1)
        x = tf.concat([x1,x2], axis=2)

        x = self.encoder(x)

        x = self.GlobalAveragePooling1D(x)
        x = self.out1(x)
        x = self.dropout(x)
        x = self.out2(x)
        shape = tf.shape(x)
        first_dim = tf.reduce_prod(shape)//(self.num_days*self.num_hid)
        x = tf.reshape(x, [first_dim, self.num_days, self.num_hid])
        return x

    

    
class Encoder_Model():
    def __init__(self, num_days) -> None:
        try:
            self.num_days = num_days
            self.model = T2VTransformer(num_days=self.num_days,num_hid=5,time_steps=10,kernel_size=3)
        except Exception as e:
            logging.info('Failed while initialising Encoder model...')
            raise CustomException(e,sys)
        
    def train(self, train_x, train_y):
        logging.info('Encoder model training started...')
        try :
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
            self.model.compile(optimizer=opt, loss='mse')

            history = self.model.fit(train_x,train_y,epochs =7)
        except Exception as e:
            logging.info('Failed while training Encoder Model...')
            raise CustomException(e,sys)

    def test(self, test_x, test_y):
        logging.info('Testing Encoder model...')
        return self.model.evaluate(test_x,test_y)
    
    def Predict(self, test_x):
        return self.model.predict(test_x)

    def prediction(self,input_sequences, targets, prediction_data):
        logging.info('Stock Price prediction started using Encoder Model...')
        try :
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
            self.model.compile(optimizer=opt, loss='mse')
            history = self.model.fit(input_sequences, targets,epochs =5)

            prediction = self.model.predict(prediction_data)
            return prediction
        except Exception as e:
            logging.info('Encoder Model failed while prediction...')
            raise CustomException(e,sys)
        