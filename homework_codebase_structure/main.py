
import sys
from preprocessing import Preprocessing
from network import My_Model
from train import Train
from test import Test
from feature import FeatureMappings

# print(sys.argv)
# if len(sys.argv == 2 and sys.argv[1] == 'test'):
#     train_bool = False
# else:
#     train_bool = True


prep = Preprocessing()  #creates preprocessing object
feature_instance = FeatureMappings(prep) #created feature object using the prep instance
train_dataset, val_dataset, test_dataset = prep.create_iterators()  # makes datasets
input_shape = (32, 16000)
num_classes = 35
network = My_Model(input_shape,num_classes)  #creates a tf.keras.Model

for i in train_dataset.take(1):
    print(i[0].shape)
    print(i[1].shape)

train_bool=True
if train_bool:
    # runs Train.py
    train_instance = Train(network, train_dataset, val_dataset)  
    train_instance.train()  #trains the model
else:
    print('test.......')
    test_obj = Test(network, test_dataset)
    test_obj.test() #tests the model
