#Making sure the TensorFlow v2 is enabled
import os
os.environ['TF2_BEHAVIOR'] = '1'
import tensorflow as tf
from tensorflow import keras
import datetime
import time

tf.random.set_seed(0)
tf.debugging.set_log_device_placement(False)

def initialize_keras_lstm(initializer, vocabulary_size, review_length,
                          embedding_size, hidden_size, dropout):
    """This function implements the LSTM model using Keras and returns the model.

    Arguments
    ---------
    initializer: function
        The weight initialization function from the torch.nn.init module that is used to initialize
        the initial weights of the models.
    vocabulary_size: int
        The number of words that are to be considered among the words that used most frequently.
    embedding_size: int
        The number of dimensions to which the words will be mapped to.
    hidden_size: int
        The number of features of the hidden state.
    dropout: float
        The dropout rate that will be considered during training.
    """
    model = keras.models.Sequential([
            keras.layers.Embedding(input_dim=vocabulary_size,
                                   output_dim=embedding_size,
                                   input_length=review_length),
        
            keras.layers.Dropout(dropout),

            keras.layers.LSTM(hidden_size, kernel_initializer=initializer,
                              recurrent_initializer='orthogonal',
                              bias_initializer='zeros', use_bias=True),

            keras.layers.Dropout(dropout),

            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
    
    return model

def keras_training_phase(model, optimizer, loss_fn, X_train_padded, y_train,
                         X_test_padded, y_test, batch_size, n_epochs, device,
                         data_type, experiment):
    """"This function implements the training phase of the Keras implementation of the LSTM
    model and returns the training time, the training timestamps (corresponding to when the training
    process began and when it ended) and the accuracy obtained on the testing dataset. The function
    also saves the model.
    
    Arguments
    ---------
    model: tf.keras.models.Sequential
        The model that is to be trained.
    optimizer: tf.keras.optimizers
        The optimizer to be used during the training process.
    loss_fn: tf.keras.losses
        The loss function that will be used during the training process
    X_train_padded: numpy array
        The training dataset that will be used to train the model.
    y_train: numpy array
        The labels of the training dataset.
    X_test_padded: numpy array
        The testing dataset that will be used to test the model.
    y_test: numpy array
        The labels of the testing dataset.
    batch_size: int
        The batch size that will be used during the training and testing processes.
    n_epochs: int
        The number of epochs for the training process.
    device: string
        The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
    data_type: string
        This string indicates whether mixed precision is to be used or not.
    experiment: string
        The string that is used to identify the model (i.e., the set of configurations the model uses).
    
    """
    if device == 'gpu':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if data_type == 'mixed':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    if data_type == 'mixed':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

    train_start_timestamp = datetime.datetime.now()
    start = time.time()
    with tf.device(device):
        model.fit(X_train_padded, y_train, batch_size=batch_size,
                             epochs=n_epochs, verbose=1)
    training_time = time.time() - start
    train_end_timestamp = datetime.datetime.now()
    
    start = time.time()
    with tf.device(device):
        accuracy = model.evaluate(X_test_padded, y_test, batch_size=batch_size)[1]
    inference_time = (time.time() - start) / X_test_padded.shape[0]
    
    #model.save('./models/lstm/{}'.format(experiment))
    
    return training_time, inference_time, accuracy * 100.0, train_start_timestamp, train_end_timestamp

def keras_inference_phase(X_test_padded_ext, y_test_ext, batch_size,
                          device, data_type, experiment):
    """This function implements the inference phase of the TensorFlow implementation of the LSTM model.
    The function returns the inference timestamps (corresponding to when the inference began and when
    it ended).
    
    Arguments
    ---------
    X_test_padded_ext: numpy array
        The larger testing dataset that is used during the inference phase.
    y_test_ext: numpy array
        The labels of the larger testing dataset.
    batch_size: int
        The batch size that will be used during the inference phase.
    device: string
        The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
    data_type: string
        This string indicates whether mixed precision is to be used or not.
    experiment: string
        The string that is used to identify the model (i.e., the set of
        configurations the model uses).
    """
    if device == 'gpu':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    if data_type == 'mixed':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
            
    model = tf.keras.models.load_model('./models/lstm/{}'.format(experiment))
    inference_start_timestamp = datetime.datetime.now()
    with tf.device(device):
        accuracy = model.evaluate(X_test_padded_ext, y_test_ext, batch_size=batch_size)[1]
    inference_end_timestamp = datetime.datetime.now()

    print('Accuracy: {}'.format(accuracy))
    return inference_start_timestamp, inference_end_timestamp