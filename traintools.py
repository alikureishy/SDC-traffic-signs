import numpy as np
import tensorflow as tf
import time
from common import *
from _datetime import timedelta

def new_weights(shape,name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)

def new_biases(length,name):
    return tf.Variable(tf.constant(0.05, shape=[length]), name=name)

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   name,
                   use_pooling=False): # Use 2x2 max-pooling.  
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape,name=name+'weights')
    biases = new_biases(length=num_filters, name=name+'biases')
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME',name=name)
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME',name=name+'pool')
    layer = tf.nn.relu(layer)
    return layer, weights

# A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer.
def flatten_layer(layer,name):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat1 = tf.reshape(layer, [-1, num_features], name=name)
    return layer_flat1, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 name,
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs],name=name+'weights')
    biases = new_biases(length=num_outputs,name=name+'biases')
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer,name=name+'relu')
    return layer

def generate_batches(data):
    batches = []
    num_batches = int(np.ceil(data.count / data.batch_size))
    for batch in range(num_batches):
        i = batch * data.batch_size
        j = i + min(data.batch_size, data.count - i)
        batches.append((i, j))
    return batches

def check_accuracy(label_predictor, data, meta, params, session, images_key, hotlabels_key, keep_key):
    tmp = data
    predictions = np.zeros(shape=data.count, dtype=np.int)
    batches = generate_batches(data)
    for batch in range(len(batches)):
        (i, j) = batches[batch]
        (test_images, hot_labels) = (tmp.images[i:j], tmp.hot_labels[i:j])
        feed_dict_test = {images_key: test_images, hotlabels_key: hot_labels, keep_key: params.dropout}
        predictions[i:j] = session.run(label_predictor, feed_dict=feed_dict_test)

    gradings = (data.labels == predictions)
    correct_sum = gradings.sum()
    acc = float(correct_sum) / data.count
    return correct_sum, acc, predictions, gradings

def train_epoch(optimizer, predictor, data, meta, params, session, images_key, hotlabels_key, keep_key):
    epoch_start_time = time.time()

    # Prepare split (it is assumed that the data is already shuffled)
    num_train_samples = data.count - params.validation_set_size # Keep the last few for validation
    print ("\t\tSplit:\tTraining = {} samples ({:.1%})\t /\t Validation = {} samples ({:.1%})".format(num_train_samples, \
                                                                                               num_train_samples/data.count,\
                                                                                               params.validation_set_size,\
                                                                                               params.validation_set_size/data.count))
    train_data = Data(\
                      images=data.images[0:num_train_samples],
                      pre_images=data.pre_images[0:num_train_samples],
                      labels=data.labels[0:num_train_samples],
                      hot_labels=data.hot_labels[0:num_train_samples],
                      count=num_train_samples,
                      batch_size=data.batch_size)
    validation_data = Data(\
                           images=data.images[num_train_samples:data.count],
                           pre_images=data.pre_images[num_train_samples:data.count],
                           labels=data.labels[num_train_samples:data.count],
                           hot_labels=data.hot_labels[num_train_samples:data.count],
                           count=params.validation_set_size,
                           batch_size=data.batch_size)
    
    train_batches = generate_batches(train_data)
    for train_batch in range(len(train_batches)):
        (i, j) = train_batches[train_batch]
        (train_images, train_hot_labels) = (train_data.images[i:j], train_data.hot_labels[i:j])
        feed_dict_train = {images_key: train_images, hotlabels_key: train_hot_labels, keep_key: params.dropout}
        session.run(optimizer, feed_dict=feed_dict_train)

        # Every 100 batches, we run the validation set:
        if train_batch % params.validation_frequency == 0:
            correct, accuracy, predictions, gradings = check_accuracy(predictor, validation_data, meta, params, session, images_key, hotlabels_key, keep_key)
            print("\t\tTraining batch: {:>6}, Validation-Accuracy: {:.1%} ({} / {})".format(train_batch+1, accuracy, correct, validation_data.count))
            #plot_example_errors(validation_data, predictions, gradings)
 
    epoch_end_time = time.time()
    epoch_time_dif = epoch_end_time - epoch_start_time
    print("\t\tEpoch time usage: " + str(timedelta(seconds=int(round(epoch_time_dif)))))
