import tensorflow as tf
from config import config
import os
class TensorboardCallback(tf.keras.callbacks.Callback):
    """
    this class is for tensorboard summaries
    implement __init__() and on_train_batch_end() functions
    you should be able to save summaries with config['train_params']['summary_step'] frequency
    tensorboard should show loss and accuracy for train and validation separately
    those in their respective folders defined in train
    """

    def __init__(self, log_dir, summary_step):
        super(TensorboardCallback, self).__init__()
        self.train_log_dir = os.path.join(log_dir, 'train')
        self.val_log_dir = os.path.join(log_dir, 'dev')
        self.summary_step = summary_step
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.step_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        # Update summaries at the end of each epoch
        if logs is not None:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', logs.get('loss'), step=epoch)
                tf.summary.scalar('accuracy', logs.get('accuracy'), step=epoch)
            with self.val_summary_writer.as_default():
                tf.summary.scalar('val_loss', logs.get('val_loss'), step=epoch)
                tf.summary.scalar('val_accuracy', logs.get('val_accuracy'), step=epoch)

class WeightsSaver(tf.keras.callbacks.Callback):
    """
    this class is for checkpoints
    implement __init__ and on_train_batch_end functions and any other auxilary functions you may need
    it should be able to save at  config['train_params']['latest_checkpoint_step']
    it should save 'max_to_keep' number of checkpoints EX. if max_to_keep = 5, you should keep only 5 newest checkpoints
    save in the folder defined in train 
    """
    def __init__(self, checkpoint_dir, latest_checkpoint_step, max_to_keep):
        super(WeightsSaver, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.latest_checkpoint_step = latest_checkpoint_step
        self.max_to_keep = max_to_keep
        self.step_counter = 0
        self.checkpoints = []

    def on_train_batch_end(self, batch, logs=None):
        self.step_counter += 1
        if self.step_counter % self.latest_checkpoint_step == 0:
            self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'ckpt_{self.step_counter}')
        self.model.save_weights(checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        # Remove old checkpoints
        if len(self.checkpoints) > self.max_to_keep:
            oldest_checkpoint = self.checkpoints.pop(0)
            os.remove(oldest_checkpoint)
