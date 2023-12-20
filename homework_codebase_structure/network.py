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

        # Define the input layer based on the input shape
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)

        # Add LSTM layers for sequence processing
        self.lstm_layer1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm_layer2 = tf.keras.layers.LSTM(32)

        # Add a dense layer for classification
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, **kwargs):
        # Define the forward pass of your model using the layers defined in __init__
        x = self.input_layer(inputs)
        x = tf.keras.layers.Reshape((32, 500))(x)

        # LSTM layers for sequence processing
        x = self.lstm_layer1(x)
        x = self.lstm_layer2(x)

        # Output layer for classification
        output = self.output_layer(x)
        return output

