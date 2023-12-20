# The dataset for this task is Goole Speech Commandc v2
# You can download it at http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

import os
from random import shuffle
import tensorflow as tf
import glob
from config import config
#dont import other libraries

class Preprocessing:
    def __init__(self):
        print('preprocessing instance creation started')

        self.dir_name = config['data_dir']
        self.input_len = config['input_len']
        # optional/bonus points: add later noise augmentation
        
    def create_iterators(self):
        # get the filenames split into train test validation
        test_files = self.get_files_from_txt('testing_list.txt')
        val_files = self.get_files_from_txt('validation_list.txt')
        filenames = glob.glob(os.path.join(self.dir_name, '*/**.wav'), recursive=True)
        filenames = [filename for filename in filenames if 'background_noise' not in filename]
        train_files = list(set(filenames) - set(val_files) - set(test_files)) #from all files subtract test and validation
        shuffle(train_files)
        # get the commands and some prints
        self.commands = self.get_commands()
        self.num_classes = len(self.commands)
        print('len(train_data)', len(train_files))
        print('prelen(test_data)', len(test_files))
        print('len(val_data)', len(val_files))
        print('commands: ', self.commands)
        print('number of commands: ', len(self.commands))

        # make tf dataset object
        self.train_dataset = self.make_tf_dataset_from_list(train_files)
        self.val_dataset = self.make_tf_dataset_from_list(val_files, is_validation = True)
        self.test_dataset = self.make_tf_dataset_from_list(test_files)

        return  self.train_dataset, self.val_dataset, self.test_dataset
    def get_files_from_txt(self, which_txt):
        """
         There are testing_list and validation_list txts and you should use those to get the the train_test_validation split
         this function must get the argument and return the paths of (for example validation) datapoints paths as a list
         you only need importet libraries
         dont forget to shuffle
        """
        assert which_txt == 'testing_list.txt' or which_txt == 'validation_list.txt', 'wrong argument'
        with open(os.path.join(self.dir_name, which_txt), 'r') as file:
            paths = file.readlines()
        paths = [os.path.join(self.dir_name, path.strip()) for path in paths]
        shuffle(paths)
        return paths

    def get_commands(self):
        dirs = glob.glob(os.path.join(self.dir_name, "*", ""))
        commands = [os.path.split(os.path.split(dir)[0])[1] for dir in dirs if 'background' not in dir]
        return commands

    
    # def get_label(self, file_path):
    #     label = file_path.split(os.sep)[0]
    #     return label
    def get_label(self, file_path):
        # Using TensorFlow's string split function
        parts = tf.strings.split(file_path, os.sep)
        # The label is assumed to be the first part after the split
        # Here, tf.strings.split returns a RaggedTensor, so we get the first item from it
        label = parts[0]
        return label

    def make_tf_dataset_from_list(self, filenames_list, is_validation = False):
        """
        ARGS:
            filenames_list is a list of file_paths
            is_validation is a boolean which should be true when makeing val_dataset

        Using the list create tf.data.Dataset object
        do necessary mappings (methods starting with 'map'),
        use prefetch, shuffle, batch methods
        bonus points` mix with background noise 
        """
        dataset = tf.data.Dataset.from_tensor_slices(filenames_list)
        dataset = dataset.map(self.map_get_waveform_and_label,
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.map_add_padding, num_parallel_calls=tf.data.AUTOTUNE)

        if not is_validation:
            dataset = dataset.shuffle(buffer_size=len(filenames_list))

        dataset = dataset.batch(config['train_params']['batch_size'])
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def map_get_waveform_and_label(self, file_path):
        """
        Map function
        for every filepath return its waveform (use only tensorflow) and label
        """
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform, _ = tf.audio.decode_wav(audio_binary)
        waveform = tf.squeeze(waveform, axis=-1)
        return waveform, label

    def map_add_padding(self, audio, label):
        return [self.add_paddings(audio), label]

    def add_paddings(self, wav):
        """
        all the data should be 2 seconds (16000 points)
        pad with zeros to make every wavs lenght 16000 if needed.
        """
        padding_needed = tf.maximum(self.input_len - tf.shape(wav)[0], 0)
        padded_wav = tf.pad(wav, [[0, padding_needed]], mode='CONSTANT',
                            constant_values=0)
        padded_wav.set_shape([self.input_len])

        return padded_wav
