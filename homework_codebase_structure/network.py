import tensorflow as tf
from config import config
import custom_layers

# config_model = config['model']

"""

Here you should implement a network 
It should be LSTM or convolutional
You can implement any thing if you can reach accuracy >85% 
It should be tf.keras.Model
you are free to use ani API
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class My_Model(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(My_Model, self).__init__()
        self.reshape = tf.keras.layers.Reshape((100, 160), input_shape=input_shape)

        self.lstm1 = LSTM(128, return_sequences=True, input_shape=input_shape)
        self.dropout1 = Dropout(config['model_params']['dropout'])
        self.lstm2 = LSTM(128)
        self.dropout2 = Dropout(config['model_params']['dropout'])
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.reshape(inputs)

        x = self.lstm1(x)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        return self.dense(x)
