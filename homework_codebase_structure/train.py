import tensorflow as tf
import callbacks
from config import config
import os
from tensorflow.keras.metrics import CategoricalAccuracy as tf_Accuracy

config_train_params = config['train_params']

#empty, only function names init and train

class Train:
    def __init__(self, model_object, train_dataset, dev_dataset):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model_object

        models_dir = './models'
        self.model_name = config['model_name']

        """
        define self.summary_dir, self.checkpoint_dir
        these are paths directories where you will store tensorboard summaries and checkpoints
        for each of your models in ./models folder should be a directory with model name, inside which should exist summaries and checpoints subdirectories
        check if they already exist and if not make them with os.makedirs        
        divide summaries into train and dev subfolders

        """
        self.checkpoint_dir = os.path.join(models_dir, self.model_name, 'checkpoints')
        self.summary_dir = os.path.join(models_dir, self.model_name, 'summaries')
        for directory in [self.checkpoint_dir, os.path.join(self.summary_dir, 'train'),
                          os.path.join(self.summary_dir, 'dev')]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    def train(self):  
        """
        In this function 
        complile self.model object (it should be a tf.keras.Model)
        use any optimizer you like, default can be adam
        find a good loss and metric for the problem

        after that check if there exist a checkpoint, if yes: restore the model


        define callbacks

        fit the model
        """
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            # Choose a loss function suitable for your problem
            metrics=[tf_Accuracy()]
        )
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.model.load_weights(latest_checkpoint)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.summary_dir,
                                                              update_freq='batch')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_dir, 'checkpoint_epoch'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

        # tboard = callbacks.LogMetricsCallback(#arguments)                                              )
        # checkpoint_callback = callbacks.WeightsSaver(#arguments)

        #fit the model
        self.model.fit(
            self.train_dataset,
            epochs=config_train_params['epochs'],
            validation_data=self.dev_dataset,
            callbacks=[tensorboard_callback, checkpoint_callback]
        )
    