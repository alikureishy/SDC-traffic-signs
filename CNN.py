
#!!get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.use('TkAgg')
from numpy import float32
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import cv2
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot
from collections import namedtuple

def crop(image, coords, size):
    # Original dimensions of the image
    org_x = size[0]
    org_y = size[1]
    
    # Scaling : Image has been scaled from its original size down to the processing size (32)
    sz_x = len(image)
    sz_y = len(image[0])
    
    # Ratio : Of actual/original (this is to scale the rectangular coordinates around image)
    rat_x = float(sz_x / org_x)
    rat_y = float(sz_y / org_y)
    
    # Rectangle: Scale the rectangular box based on the scaling above
    x1 = rat_x * coords[0]
    x2 = rat_x * coords[2]
    y1 = rat_y * coords[1]
    y2 = rat_y * coords[3]

    cropped = image[x1:x2, y1:y2]

    return cropped

def transform_image(image, ang_range, shear_range, trans_range):
    '''
    This function transforms input_images to generate new input_images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = image.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    image = cv2.warpAffine(image,Rot_M,(cols,rows))
    image = cv2.warpAffine(image,Trans_M,(cols,rows))
    image = cv2.warpAffine(image,shear_M,(cols,rows))
    
    return image

def sharpen_blur(image):
    return image

def rotate(image):
    return image

def brighten_darken(image):
    return image

def push_away(image, newshape):
    return image

def change_perspective(image):
    return image

def add_noise(image, noise):
    return image

def resize(image, size):
    if size==image.shape[0] and size==image.shape[1]:
        return image
    
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the dimensions of the new image to the old image
    r_x = size / image.shape[0]
    r_y = size / image.shape[1]
    dim = (int(image.shape[0] * r_x), int(image.shape[1] * r_y))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def equalize_distribution(xs, ys, coords, sizes):
    more_images = []
    more_labels = []
    
    (num_images, img_size, _, num_channels) = xs.shape

    (unique, counts) = np.unique(ys, return_counts=True)
    maxcount = max(counts)
    mincount = min(counts)
    delta_threshold = 25 # percent
    
    n_classes = len(unique) # Now we will assume that the uniques are 0..n (strictly increasing and continuous)
    # Calculate inverse lookup
    inverse = np.empty((n_classes, 0)).tolist()
    for i in range(len(xs)):
        inverse[ys[i]].append(i)

    if (100 - (mincount * 100 / maxcount)) > delta_threshold:
        idealcount = maxcount #int(maxcount + .25 * maxcount)
        descending = np.argsort(counts)[::-1] # Sort the classes in descending order based on 'counts' array values
        for cls in descending:
            needed = idealcount - counts[cls]
            print ("Class\t{}:\t{} examples \t[% = {}\tNeed = {} more]".format(cls, counts[cls], counts[cls]*100/num_images, needed))
            
            # Perform a sequence of modifications to generate the 'needed' new data, using
            # a uniform distribution for the actual modifications
            if (needed > 0):
                images = [xs[i] for i in inverse[cls]]  # Get the exact indices of each image for the current class

                for idx in range(needed):
                    i = idx % len(images)
                    image = images[i]
                    # Can look at: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb
                    #   for ideas for image transformations
                    #image = crop(image, coords[i], sizes[i])
                    #image = addnoise(image, 5)
                    #image = rotate(image)
                    #image = sharpen_blur(image)
                    #image = push_away(image, (img_size, img_size, num_channels))
                    #image = change_perspective(image)
                    image = transform_image(image, 20, 10, 5)
                    image = resize(image, img_size) # Get image back to required size, in case that changed
                    
                    more_images.append(image)
                    more_labels.append(cls)
        
    return more_images, more_labels

# Function used to plot 9 input_images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(images, actual_labels, label_predictor=None):
    assert len(images) == len(actual_labels) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if label_predictor is None:
            xlabel = "True: {0}".format(actual_labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(actual_labels[i], label_predictor[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# ## Load Data
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
training_file = 'data/train.p'
testing_file = 'data/test.p'
more_data_file = 'data/more_data.p'
checkpoint_file = 'checkpoints/checkpoint.chk'
abort_file = 'abort'

FEATURES = 'features'
LABELS = 'labels'
SIZES = 'sizes'
COORDS = 'coords'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train, sizes_train, coords_train = train[FEATURES], train[LABELS], train[SIZES], train[COORDS]
x_test, y_test, sizes_test, coords_test = test[FEATURES], test[LABELS], test[SIZES], test[COORDS]

img_size = len(x_train[0])
num_channels = len(x_train[0][0][0])
img_shape = x_train[0].shape #(img_size, img_size, num_channels)

# Determining augmentation
augment_data = True
if augment_data:
    print ("Processing augmentation...")
    if not os.path.isfile(more_data_file):
        print ("\tGenerating fresh data...")
        more_images, more_labels = equalize_distribution(x_train, y_train, coords_train, sizes_train)
        if (len(more_images)>0):
            print('\t\tSaving data to pickle file...')
            try:
                with open(more_data_file, 'wb') as pfile:
                    pickle.dump(
                        {
                            FEATURES: more_images,
                            LABELS: more_labels,
                        },
                        pfile, pickle.HIGHEST_PROTOCOL)
                print ('\t\tGenerated and saved ({} input_images) for subsequent use.'.format(len(more_images)))
            except Exception as e:
                print('\t\tUnable to save data to', more_data_file, ':', e)
                raise
    else:
        print ("\tData already exists.")
    
    with open(more_data_file, 'rb') as f:
        print ("\tReading in augmented data...")
        more = pickle.load(f)
        x_more, y_more = more[FEATURES], more[LABELS]
        print ("\t\tAugmented data count:\t{}".format(len(x_more)))
        #plot_images(x_more[0:9], actual_labels=y_more[0:9])
        x_train = np.concatenate((x_train, x_more))
        y_train = np.concatenate((y_train, y_more))

# Free up some memory
del sizes_train, sizes_test, coords_train, coords_test

# Normalize the input_images (and save originals)
x_train_norm = np.zeros_like(x_train, dtype=float32)
x_test_norm = np.zeros_like(x_test, dtype=float32)
for i in range(0, len(x_train)):
    cv2.normalize(x_train[i], x_train_norm[i], 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
for i in range(0, len(x_test)):
    cv2.normalize(x_test[i], x_test_norm[i], 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

num_classes = len(np.unique(y_train))
y_train_hot = dense_to_one_hot(y_train, num_classes) # One-hot encode all the labels
y_test_hot = dense_to_one_hot(y_test, num_classes) # One-hot encode all the labels

# Define container for all data related to training/testing
Data = namedtuple ('Data', ['images', 'pre_images', 'labels', 'hot_labels', 'count', 'batch_size'])
Meta = namedtuple ('Meta', ['image_shape', 'num_channels', 'num_classes'])
Params = namedtuple ('Params', ['num_train_epochs', \
                                'learning_rate', \
                                'dropout', \
                                'validation_set_size', \
                                'validation_frequency', \
                                'training_accuracy_threshold', \
                                'do_checkpointing'])

train = Data(x_train_norm, x_train, y_train, y_train_hot, len(x_train), batch_size=128) # Use the normalized input_images (x_train_norm)
test = Data(x_test_norm, x_test, y_test, y_test_hot, len(x_test), batch_size=256) # Use the normalized input_images (x_test_norm)
meta = Meta(x_test_norm[0].shape, len(x_test[0][0][0]), num_classes)
params = Params (\
                 num_train_epochs=0,
                 learning_rate=1e-3,
                 dropout=0.5, # this is actually keep_prob
                 validation_set_size=int(0.10 * train.count),
                 validation_frequency=100, # Validation is performed every 'n' batches
                 training_accuracy_threshold=0.95, \
                 do_checkpointing=True)

print("Size of:")
print("- Training-set:\t\t{}".format(train.count))
print("- Test-set:\t\t{}".format(test.count))
print("- Shape:\t\t{}".format(meta.image_shape))
print("- Num channels:\t{}".format(meta.num_channels))
print("- Num classes:\t{}".format(meta.num_classes))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=False):  # Use 2x2 max-pooling.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

# A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer.
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

def generate_batches(data):
    batches = []
    num_batches = int(np.ceil(data.count / data.batch_size))
    for batch in range(num_batches):
        i = batch * data.batch_size
        j = i + min(data.batch_size, data.count - i)
        batches.append((i, j))
    return batches

# Shuffle the training and testing data:
def shuffle (data):
    seq = np.arange(0, len(data.images))
    np.random.shuffle(seq)
    images = np.array([ data.images[i] for i in seq])
    pre_images = np.array([ data.pre_images[i] for i in seq])
    labels = np.array([ data.labels[i] for i in seq])
    hot_labels = np.array([ data.hot_labels[i] for i in seq])
    return Data(images=images, pre_images=pre_images, labels=labels, hot_labels=hot_labels, count=len(images), batch_size=data.batch_size)

def check_accuracy(label_predictor, data, meta, params):
    tmp = data
    predictions = np.zeros(shape=data.count, dtype=np.int)
    batches = generate_batches(data)
    for batch in range(len(batches)):
        (i, j) = batches[batch]
        (test_images, hot_labels) = (tmp.images[i:j], tmp.hot_labels[i:j])
        feed_dict_test = {images: test_images, actual_hot_labels: hot_labels, keep_prob: params.dropout}
        predictions[i:j] = session.run(label_predictor, feed_dict=feed_dict_test)

    gradings = (data.labels == predictions)
    correct_sum = gradings.sum()
    acc = float(correct_sum) / data.count    
    return correct_sum, acc, label_predictor, gradings

def train_epoch(optimizer, predictor, data, meta, params):
    epoch_start_time = time.time()

    # Prepare split (it is assumed that the data is already shuffled)
    num_train_samples = data.count - params.validation_set_size # Keep the last few for validation
    print ("\t\tSplit:\tTraining = {} samples\t /\t Validation = {} samples ({:.1%}) split".format(num_train_samples, \
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
        feed_dict_train = {images: train_images, actual_hot_labels: train_hot_labels, keep_prob: params.dropout}
        session.run(optimizer, feed_dict=feed_dict_train)

        # Every 100 batches, we run the validation set:
        if train_batch % params.validation_frequency == 0:
            correct, accuracy, predictions, gradings = check_accuracy(predictor, validation_data, meta, params)
            print("\t\tTraining batch: {:>6}, Validation-Accuracy: {:.1%} ({} / {})".format(train_batch+1, accuracy, correct, validation_data.count))
            #plot_example_errors(validation_data, predictions, gradings)
 
    epoch_end_time = time.time()
    epoch_time_dif = epoch_end_time - epoch_start_time
    print("\t\tEpoch time usage: " + str(timedelta(seconds=int(round(epoch_time_dif)))))

# Function for plotting examples of input_images from the test-set that have been mis-classified.
def plot_example_errors(data, predictions, gradings):
    incorrect = (gradings == False)
    images = data.images[incorrect] #images = [ data.images[i] for i in range(data.count) where incorrect[i]]
    predictions = predictions[incorrect]
    actuals = data.labels[incorrect]
    plot_images(images=images[0:9], actual_labels=actuals[0:9], predictions=predictions[0:9])

# Helper-function for plotting an image.
def plot_image(image):
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    plt.show()

### End of methods

#################################
#          PIPELINE             #
#################################

# Convolutional layers:
filter_size1 = 5          # Convolution filters are 5 input_images 5 pixels.
num_filters1 = 32         # There are 16 of these filters.

filter_size2 = 5          # Convolution filters are 5 input_images 5 pixels.
num_filters2 = 64         # There are 36 of these filters.

filter_size3 = 5          # Convolution filters are 5 input_images 5 pixels.
num_filters3 = 128         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 1024             # Number of neurons in fully-connected layer.

# Plot some training samples
#plot_images(images=train.pre_images[0:9], actual_labels=train.labels[0:9])

images = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='images')
actual_hot_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='actual_hot_labels')
actual_labels = tf.argmax(actual_hot_labels, dimension=1)
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Network architecture/layers
layer, _ = None, None
layer, _ = layer_conv1, weights_conv1 = new_conv_layer(input=images, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
#layer = layer_dropout1 = tf.nn.dropout(layer, keep_prob)

layer, _ = layer_conv2, weights_conv2 = new_conv_layer(input=layer, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
#layer = layer_dropout2 = tf.nn.dropout(layer, keep_prob)

layer, _ = layer_conv3, weights_conv3 = new_conv_layer(input=layer, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3, use_pooling=True)
#layer = layer_dropout3 = tf.nn.dropout(layer, keep_prob)

layer, _ = layer_flat, num_features = flatten_layer(layer)

layer = layer_fc1 = new_fc_layer(input=layer, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
layer = layer_dropout4 = tf.nn.dropout(layer, keep_prob)

layer = logits = layer_fc2 = new_fc_layer(input=layer, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)
#layer = logits = layer_dropout5 = tf.nn.dropout(layer, keep_prob)

hot_label_predictor = tf.nn.softmax(logits)
label_predictor = tf.argmax(hot_label_predictor, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actual_hot_labels)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(cost)
#correct_prediction = tf.equal(label_predictor, actual_labels)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print ("Intializing session:")
saver = tf.train.Saver()
session = tf.Session()
if ((not saver is None) and (os.path.isfile(checkpoint_file))):
    print ("\tUsing checkpoint.")
    saver.restore(session, checkpoint_file)
else:
    print ("\tUsing default initialization.")
    session.run(tf.initialize_all_variables())

# Now start the training epochs
unlimited_epochs = params.num_train_epochs == None
if (unlimited_epochs or params.num_train_epochs > 0):
    keep_going = True
    epoch = 0
    while keep_going:
        if os.path.isfile(abort_file):
            print ("\tAbort file found. Aborting training. [To continue, re-run training.]")
            os.remove(abort_file)
            break
    
        print ("\tTraining epoch: {}".format(epoch+1))
        train = shuffle(train)
        train_epoch(optimizer, label_predictor, train, meta, params)
    
        if (params.do_checkpointing):
            print ("\t\tSaving checkpoint.")
            saver.save(session, checkpoint_file)
        else:
            print ("\t\tCheckpointing disabled.")
            
        test = shuffle(test)
        correct, accuracy, predictions, gradings = check_accuracy(label_predictor, test, meta, params)
        print("\t\t\tAccuracy on Test-Set: {0:.1%} ({1} / {2})".format(accuracy, correct, test.count))
            
        if accuracy >= params.training_accuracy_threshold:
            print("\t\t\tAchieved threshold accuracy! Ending training.")
            break
        epoch += 1
        keep_going = unlimited_epochs or epoch < params.num_train_epochs
    
    test = shuffle(test)
    correct, accuracy, predictions, gradings = check_accuracy(label_predictor, test, meta, params)
    print("Final accuracy on Test-Set: {0:.1%} ({1} / {2})".format(accuracy, correct, test.count))


print ("\nDONE!")
#plot_example_errors(test, predictions, gradings)

#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=0.5).minimize(cost)
#optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.0).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost)

