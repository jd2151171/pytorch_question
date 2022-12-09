import torch
from torch.nn import Module
from torch import nn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt
import random
import time
import csv
import tensorflow as tf
import datetime
import os
import sys


implemented_frameworks = ['PyTorch', 'Keras', 'TensorFlow']
implemented_devices = ['cpu', 'gpu']


i = 0
training_size = 1
batch_size = 256
# if len(sys.argv) > 3:
#     n_epochs = int(sys.argv[3])
# else:
n_epochs = 5
learning_rate = 0.01
data_type = 'float32'
device = sys.argv[2]
weight_initialization = 'xavier'
framework = sys.argv[1]
dropout = 0.15
phase = 'training'

if framework not in implemented_frameworks:
    raise Exception('{} not recognized. Please choose one of the available frameworks (i.e., PyTorch, Keras and TensorFlow)'.format(framework))

if device not in implemented_devices:
    raise Exception('{} not recognized. Please choose one of the available devices (i.e., cpu and gpu)'.format(device))
    
    

experiment = 'lstm_{}{}_{}ts_{}batch_{}epochs_{}lr_{}dtype_{}_{}wi_{}dp'.format(framework, i, training_size,
                                                                                batch_size, n_epochs,
                                                                                learning_rate, data_type, device,
                                                                                weight_initialization, dropout)


#making sure tensorflow/keras do not reserve all the VRAM available on GPU
if device == 'gpu':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
def load_and_preprocess_data(training_size, vocab_size=10000, review_length=500):
    """This function loads and preprocesses (i.e., pads) the IMDB movie review dataset. The
    function returns the training dataset (according to the passed training size),
    the testing dataset and the larger testing dataset (i.e., the dataset that will be used during
    the inference phase).
    
    
    Arguments
    ---------
    training_size: float
        The proportion of the original training dataset that is used during the training process.
    vocab_size: int
        The number of words that are to be considered among the words that used most frequently.
    review_lenght: int
        The maximum lenght of the movie reviews loaded.
    """
    #Loading the IMDB movie review dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

    #Setting the training set to correspond to the training size in the configurations
    X_train = X_train[:int(training_size*X_train.shape[0])]
    y_train = y_train[:X_train.shape[0]]

    #Setting up the larger dataset that will be used during the inference phase.
    X_test_ext = X_test.copy()
    y_test_ext = y_test.copy()
    for j in range(8):
        X_test_ext = np.append(X_test_ext, X_test.copy(), axis=0)
        y_test_ext = np.append(y_test_ext, y_test.copy(), axis=0)

    #Padding the reviews so they are all the same lenght
    X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train, maxlen = review_length)
    X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test, maxlen = review_length)
    X_test_padded_ext = keras.preprocessing.sequence.pad_sequences(X_test_ext, maxlen = review_length)
    
    return X_train_padded, y_train, X_test_padded, y_test, X_test_padded_ext, y_test_ext



collect_time = 0




data_type_list = ['float32', 'mixed']

device_dict = {
    'cpu': {
        'PyTorch': 'cpu',
        'Keras': '/cpu:0',
        'TensorFlow': '/CPU:0'
    },
    'gpu': {
        'PyTorch': 'cuda',
        'Keras': '/gpu:0',
        'TensorFlow': '/GPU:0'
    }
}

weight_initialization_dict = { 
    'xavier': {
        'PyTorch': torch.nn.init.xavier_normal_,
        'Keras': tf.keras.initializers.GlorotNormal,
        'TensorFlow': tf.compat.v1.initializers.glorot_normal
    },
    'he': {
        'PyTorch': torch.nn.init.kaiming_normal_,
        'Keras': tf.keras.initializers.HeNormal,
        'TensorFlow': tf.compat.v1.keras.initializers.he_normal
    }
}


# if not os.path.isdir('./models/lstm/{}'.format(experiment)):
#     os.mkdir('./models/lstm/{}'.format(experiment))

#we try to minimize the randomness as much as possible
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
# tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)



print('Training {}'.format(experiment))

training_time = 0
inference_time = 0
accuracy = 0

train_start_timestamp = 0
train_end_timestamp = 0

inference_start_timestamp = 0
inference_end_timestamp = 0

vocab_size = 10000
review_length = 500

embedding_size = 32
hidden_size = 100

#Loading the IMDB dataset
X_train_padded, y_train, X_test_padded, y_test, X_test_padded_ext, y_test_ext = load_and_preprocess_data(training_size,
                                                                                                         vocab_size=10000,
                                                                                                         review_length=500)

print('Training {}'.format(experiment))

training_time = 0
inference_time = 0
accuracy = 0

train_start_timestamp = 0
train_end_timestamp = 0

inference_start_timestamp = 0
inference_end_timestamp = 0



if framework == 'PyTorch':
    import pytorch_lstm as pytorch_lstm
    
#     num_threads = -1
    
#     if len(sys.argv) > 3:
#         num_threads = int(sys.argv[3])
    print(*torch.__config__.show().split("\n"), sep="\n")
    
    pytorch_train_loader, pytorch_test_loader, pytorch_test_loader_ext = pytorch_lstm.generate_pytorch_dataloader(X_train_padded, X_test_padded, X_test_padded_ext, y_train, y_test, y_test_ext, batch_size, device_dict[device][framework], review_length=500)

    model = pytorch_lstm.PyTorchLSTMMod(weight_initialization_dict[weight_initialization][framework],
                                        vocab_size, embedding_size, hidden_size, dropout)
    model = model.to(device_dict[device][framework])
    
    if phase == 'training':
        from torch.optim import Adam
        
        optimizer = Adam(model.parameters(), lr=learning_rate)
        training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp = pytorch_lstm.pytorch_training_phase(model, optimizer,
                                                                                                                                  pytorch_train_loader, pytorch_test_loader,
                                                                                                                                  n_epochs, device_dict[device][framework],
                                                                                                                                  data_type, experiment)
    elif phase == 'inference':
        inference_start_timestamp, inference_end_timestamp = pytorch_lstm.pytorch_inference_phase(model, experiment, pytorch_test_loader_ext,
                                                                                                  device_dict[device][framework], data_type)
    
    #We take the mean time the model takes to infer a single sample.
    inference_time /= X_test_padded.shape[0]


if framework == 'Keras':
    os.environ['TF2_BEHAVIOR'] = '1'
    import tensorflow as tf

    if device == 'gpu':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
    tf.random.set_seed(0)
    
    import keras_lstm as keras_lstm
    tf.debugging.set_log_device_placement(False)
#     if data_type == 'mixed':
#         policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#         tf.keras.mixed_precision.experimental.set_policy(policy)
    
    if phase == 'training':
    
        model = keras_lstm.initialize_keras_lstm(weight_initialization_dict[weight_initialization][framework],
                                                 vocab_size, review_length, embedding_size, hidden_size,
                                                 dropout)
    
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.binary_crossentropy
        
        training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp = keras_lstm.keras_training_phase(model, optimizer,
                                                                                                                              loss_fn, X_train_padded,
                                                                                                                              y_train, X_test_padded,
                                                                                                                              y_test, batch_size, n_epochs,
                                                                                                                              device_dict[device][framework],
                                                                                                                              data_type, experiment)
    elif phase == 'inference':
        inference_start_timestamp, inference_end_timestamp = keras_lstm.keras_inference_phase(X_test_padded_ext, y_test_ext,
                                                                                              batch_size, device_dict[device][framework],
                                                                                              data_type, experiment)

if framework == 'TensorFlow':
    #the first version of tensorflow needs to be used, as tensorflow 2.0 uses keras by default
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    tf.compat.v1.set_random_seed(0)
    
    import tensorflow_lstm as tensorflow_lstm

    tf.debugging.set_log_device_placement(False)
    
    with tf.device(device_dict[device][framework]):
        with tf.compat.v1.variable_scope(name_or_scope='TensorFlowLSTM', reuse=tf.compat.v1.AUTO_REUSE,
                                         initializer=weight_initialization_dict[weight_initialization][framework]):

            model = tensorflow_lstm.TensorFlowLSTMMod(weight_initialization_dict[weight_initialization][framework], vocab_size,
                                                      embedding_size, hidden_size, dropout, device_dict[device][framework])
            
            #Collecting the lenghts of the sequences (note that all the sequences are of the same length as they
            #have been padded).
            lens_train = np.array([len(xi) for xi in X_train_padded], dtype='int32')
            lens_test = np.array([len(xi) for xi in X_test_padded], dtype='int32')
            lens_test_ext = np.array([len(xi) for xi in X_test_padded_ext], dtype='int32')
            if phase == 'training':
            
                training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp = tensorflow_lstm.tensorflow_training_phase(model, learning_rate,
                                                                                                                                                review_length,
                                                                                                                                                X_train_padded,
                                                                                                                                                lens_train, y_train,
                                                                                                                                                X_test_padded, 
                                                                                                                                                lens_test, y_test, 
                                                                                                                                                batch_size, n_epochs,
                                                                                                                                                device_dict[device][framework],
                                                                                                                                                data_type, experiment)
            elif phase == 'inference':
                
                inference_start_timestamp, inference_end_timestamp = tensorflow_lstm.tensorflow_inference_phase(model, review_length,
                                                                                                                X_test_padded_ext,
                                                                                                                lens_test_ext, y_test_ext,
                                                                                                                batch_size,
                                                                                                                device_dict[device][framework],
                                                                                                                data_type, experiment)

#Writing the results collected during the training or the inference phase
if phase == 'training':
    results = {
        'training_time': training_time,
        'inference_time': inference_time,
        'accuracy': accuracy,
        'train_start_timestamp': train_start_timestamp,
        'train_end_timestamp': train_end_timestamp
    }

elif phase == 'inference':
    results = {
        'inference_start_timestamp': inference_start_timestamp,
        'inference_end_timestamp': inference_end_timestamp
    }

print(results)