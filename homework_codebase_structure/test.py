import os
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy as tf_Accuracy
from config import config
 #empty
#only comments
class Test:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def test(self):

        model_name = config['model_name']
        models_dir = './models'
        checkpoint_dir = os.path.join(models_dir, model_name, 'checkpoints')

        if os.listdir(checkpoint_dir):
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                self.model.load_weights(latest_checkpoint)
                print(f"Restored model from checkpoint: {latest_checkpoint}")
            else:
                print("No checkpoint found. Testing with untrained model.")

        else:
            print("Checkpoint directory is empty. Testing with untrained model.")
            return
        loss, accuracy = self.model.evaluate(self.dataset)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")





