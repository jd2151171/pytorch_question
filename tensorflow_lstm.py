"""TensorFlow (v1) implementation of the LSTM model."""

#the first version of tensorflow needs to be used, as tensorflow 2.0 uses keras by default
import tensorflow.compat.v1 as tf

import datetime
import time
import os

tf.disable_v2_behavior()

tf.compat.v1.set_random_seed(0)

class TensorFlowLSTMMod:
    """This class implements the LSTM model using TensorFlow.

    Arguments
    ---------
    initializer: function
        The weight initialization function that is used to initialize the initial weights of
        the models.
    vocabulary_size: int
        The number of words that are to be considered among the words that used most frequently.
    embedding_size: int
        The number of dimensions to which the words will be mapped to.
    hidden_size: int
        The number of features of the hidden state.
    dropout: float
        The dropout rate that will be considered during training.
    """
    def __init__(self, initializer, vocabulary_size, embedding_size, hidden_size, dropout, device):
        
        self.device = device
        
        self.dropout = 1.0 - dropout
        
        self.embed = tf.get_variable('embed', (vocabulary_size, embedding_size))

        if '/GPU' in device: 
            self.lstm = tf.compat.v1.keras.layers.CuDNNLSTM(hidden_size, kernel_initializer=initializer,
                                                            recurrent_initializer='orthogonal', bias_initializer='zeros')
        else:
            self.lstm = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer)

        self.fc = tf.layers.Dense(units=1)

    def __call__(self, x, lens, is_training=False):
        """This function implements the forward pass of the model.

        Arguments
        ---------
        inputs: Tensor
            The set of samples the model is to infer.
        lens: Tensor
            The set of the length of the samples (i.e., the length of the
            sequences representing the reviews).
        device: string
            The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
        is_training: boolean
            This indicates whether the forward pass is occuring during training
            (i.e., if we should consider dropout).
        """
        x = tf.nn.embedding_lookup(self.embed, x)

        if is_training:
            x = tf.compat.v1.layers.dropout(x, self.dropout)

        if '/GPU' in self.device:
            h = self.lstm(x)
        else:
            o, (c, h) = tf.nn.dynamic_rnn(self.lstm, x, sequence_length=lens, dtype=tf.float32)
        if is_training:
            h = tf.compat.v1.layers.dropout(h, self.dropout)
        p = self.fc(h)
        return tf.reshape(p, [-1])
    
    def train_tensorflow(self, sess, training_operation, x, lens, y,
                         X_train_padded, lens_train, y_train,
                         batch_size, n_epochs):
        """This function implements the training process of the TensorFlow model.

        Arguments
        ---------
        self: TensorFlowLSTMMod
            The model that is to be trained.
        sess: tf.session
            The session that is used to run the operations of the training phase.
        training_operation: function
            The operation used to compute the loss and update the weights of the model during the
            training process.
        x: tf.placeholder
            The placeholder used for the inputs (i.e., the reviews).
        lens: tf.placeholder
            The placeholder used for the lengths of the inputs (i.e., the length of the sequences
            that represent the review).
        y: tf.placeholder
            The placeholder used for the labels.
        X_train_padded: numpy array
            The training dataset that will be used to train the model.
        lens_train: numpy array
            The set of the length of the training samples (i.e., the length of the
            sequences representing the reviews).
        y_train: numpy array
            The labels of the training dataset.
        batch_size: int
            The batch size that will be used during the training process.
        n_epochs: int
            The number of epochs for the training process.
        """

        num_examples = len(X_train_padded)
        for i in range(1, n_epochs + 1):
            epoch_start = time.time()
            count = 0
            losses_train = []

            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_lens, batch_y = X_train_padded[offset:end], lens_train[offset:end], y_train[offset:end].astype('int32')
                sess.run(training_operation, feed_dict={x: batch_x, lens: batch_lens, y: batch_y})

            print("-TensorFlow: Epoch {} done in {}s".format(i, (time.time() - epoch_start)))
    
    
    def test_tensorflow(self, sess, testing_operation, x, lens,
                        y, X_test_padded, lens_test, y_test,
                        batch_size):
        """This function implements the testing process of the TensorFlow model and returns the accuracy
        obtained on the testing dataset.

        Arguments
        ---------
        self: TensorFlowLSTMMod
            The model that is to be trained.
        sess: tf.session
            The session that is used to run the operations of the training phase.
        training_operation: function
            The operation used to compute the loss and update the weights of the model during the
            training process.
        x: tf.placeholder
            The placeholder used for the inputs (i.e., the reviews).
        lens: tf.placeholder
            The placeholder used for the lengths of the inputs (i.e., the length of the sequences
            that represent the review).
        y: tf.placeholder
            The placeholder used for the labels.
        X_testing_padded: numpy array
            The training dataset that will be used to testing the model.
        lens_testing: numpy array
            The set of the length of the testinging samples (i.e., the length of the
            sequences representing the reviews).
        y_testing: numpy array
            The labels of the testinging dataset.
        batch_size: int
            The batch size that will be used during the testing process.
        n_epochs: int
            The number of epochs for the training process.
        """
        
        outputs_test = self(x, lens)
        correct_prediction = tf.equal(tf.cast(outputs_test >= 0.0, tf.int32), y)
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        count = 0
        num_examples = len(X_test_padded)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_lens, batch_y = X_test_padded[offset:end], lens_test[offset:end], y_test[offset:end].astype('int32')
            _, preds = sess.run(testing_operation, feed_dict={x: batch_x, lens: batch_lens, y: batch_y})
            for pred in preds:
                if pred == True:
                    count = count + 1
        return (count / num_examples) * 100.0
        

def tensorflow_training_phase(model, learning_rate, review_length, X_train_padded,
                              lens_train, y_train, X_test_padded, lens_test,
                              y_test, batch_size, n_epochs, device, data_type,
                              experiment):
    """"This function implements the training phase of the TensorFlow implementation of the LSTM
    model and returns the training time, the training timestamps (corresponding to when the training
    process began and when it ended) and the accuracy obtained on the testing dataset. The function
    also saves the model.
    
    Arguments
    ---------
    model: TensorFlowLSTMMod
        The model that is to be trained.
    learning_rate: float
        The learning rate used to train the model.
    review_lenght: int
        The maximum lenght of the movie reviews loaded.
    X_train_padded: numpy array
        The training dataset that will be used to train the model.
    lens_train: numpy array
        The set of the length of the training samples (i.e., the length of the
        sequences representing the reviews).
    y_train: numpy array
        The labels of the training dataset.
    X_test_padded: numpy array
        The testing dataset that will be used to test the model.
    lens_test: numpy array
        The set of the length of the testing samples (i.e., the length of the
        sequences representing the reviews).
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
    
    #Initializaing the placeholders.
    x = tf.compat.v1.placeholder(tf.int32, (None, review_length))
    lens = tf.compat.v1.placeholder(tf.int32, (None))
    y = tf.compat.v1.placeholder(tf.int32, (None))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    outputs_train = model(x, lens, is_training=True)
    cross_entropy_train = tf.compat.v1.losses.sigmoid_cross_entropy(y, outputs_train)
    training_operation = optimizer.minimize(cross_entropy_train)
    
    outputs_test = model(x, lens, is_training=False)
    cross_entropy_test = tf.compat.v1.losses.sigmoid_cross_entropy(y, outputs_test)
    correct_prediction = tf.equal(tf.cast(outputs_test >= 0.0, tf.int32), y)
    testing_operation = [cross_entropy_test, correct_prediction]

    #If using GPU-acceleration, allow soft placement as some operations do not have a GPU kernel
    if '/GPU' in device:
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        #If using GPU-acceleration, prevent TensorFlow from reserving the entirety of the GPU's VRAM
        config.gpu_options.allow_growth=True
    else:
        config = tf.ConfigProto(log_device_placement=True)
    
    if data_type == 'mixed':
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        
    if data_type == 'mixed':
        config.graph_options.rewrite_options.auto_mixed_precision = 1
        
    with tf.compat.v1.Session(config=config) as sess:

        #Training the model
        sess.run(tf.compat.v1.global_variables_initializer())
        train_start_timestamp = datetime.datetime.now()
        start = time.time()
        model.train_tensorflow(sess, training_operation, x, lens, y, X_train_padded,
                               lens_train, y_train, batch_size, n_epochs)
        training_time = time.time() - start
        train_end_timestamp = datetime.datetime.now()
        
        #Testing the model
        start = time.time()
        accuracy = model.test_tensorflow(sess, testing_operation, x, lens, y, X_test_padded,
                                         lens_test, y_test, batch_size)
        inference_time = (time.time() - start) / X_test_padded.shape[0]
        
        #Saving the model
        #saver = tf.train.Saver()
        #saver.save(sess, './models/lstm/{}/model.ckpt'.format(experiment))
        
        print(accuracy)
        return training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp
    
def tensorflow_inference_phase(model, review_length, X_test_padded_ext, lens_test_ext,
                               y_test_ext, batch_size, device, data_type, experiment):
    """This function implements the inference phase of the TensorFlow implementation of the LSTM model.
    The function returns the inference timestamps (corresponding to when the inference began and when
    it ended).
    
    Arguments
    ---------
    model: TensorFlowLSTMMod
        The model that is to be evaluated (the model acts as a placeholder into
        which the weights of the trained model will be loaded).
    review_lenght: int
        The maximum lenght of the movie reviews loaded.
    X_test_padded_ext: numpy array
        The larger testing dataset that is used during the inference phase.
    lens_test_ext: numpy array
        The set of the length of the samples in the larger dataset (i.e.,
        the length of the sequences representing the reviews).
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
    #Initializaing the placeholders.
    x = tf.compat.v1.placeholder(tf.int32, (None, review_length))
    lens = tf.compat.v1.placeholder(tf.int32, (None))
    y = tf.compat.v1.placeholder(tf.int32, (None))
    
    outputs_test = model(x, lens, is_training=False)
    cross_entropy_test = tf.compat.v1.losses.sigmoid_cross_entropy(y, outputs_test)
    correct_prediction = tf.equal(tf.cast(outputs_test >= 0.0, tf.int32), y)
    testing_operation = [cross_entropy_test, correct_prediction]

    #If using GPU-acceleration, allow soft placement as some operations do not have a GPU kernel
    if '/GPU' in device:
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        #If using GPU-acceleration, prevent TensorFlow from reserving the entirety of the GPU's VRAM
        config.gpu_options.allow_growth=True
    else:
        config = tf.ConfigProto(log_device_placement=True)
    
    if data_type == 'mixed':
        config.graph_options.rewrite_options.auto_mixed_precision = 1

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.train.Saver()

        #Restoring the weights of the trained model
        vars_in_checkpoint = tf.train.list_variables(os.path.join('./models/lstm/{}'.format(experiment)))
        saver.restore(sess, './models/lstm/{}/model.ckpt'.format(experiment))

        inference_start_timestamp = datetime.datetime.now()
        accuracy = model.test_tensorflow(sess, testing_operation, x, lens, y, X_test_padded_ext,
                                         lens_test_ext, y_test_ext, batch_size)
        inference_end_timestamp = datetime.datetime.now()
        
        return inference_start_timestamp, inference_end_timestamp