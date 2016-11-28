
#!!get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.use('TkAgg')
from numpy import float32
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import cv2
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot

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
    This function transforms images to generate new images.
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
#                images = [crop(xs[i], coords[i], sizes[i]) for i in inverse[cls]]  # Get the exact indices of each image for the current class
                images = [xs[i] for i in inverse[cls]]  # Get the exact indices of each image for the current class

                for idx in range(needed):
                    i = idx % len(images)
                    image = images[i]
                    #image = crop(image, coords[i], sizes[i])
                    #image = rotate(image)
                    #image = sharpen_blur(image)
                    #image = push_away(image, (img_size, img_size, num_channels))
                    #image = change_perspective(image)
                    image = transform_image(image,20,10,5)
                    image = resize(image, img_size) # Get image back to required size, in case that changed
                    
                    more_images.append(image)
                    more_labels.append(cls)
        
    return more_images, more_labels

# Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# ## Load Data
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
training_file = '/Users/safdar/Datasets/self-driving-car/traffic-signs-data/train.p'
testing_file = '/Users/safdar/Datasets/self-driving-car/traffic-signs-data/test.p'
more_data_file = '/Users/safdar/Datasets/self-driving-car/traffic-signs-data/more_data.p'

FEATURES = 'features'
LABELS = 'labels'
SIZES = 'sizes'
COORDS = 'coords'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train, sizes_train, coords_train = train[FEATURES], train[LABELS], train[SIZES], train[COORDS]
#del train # free up memory
x_test, y_test, sizes_test, coords_test = test[FEATURES], test[LABELS], test[SIZES], test[COORDS]
#del test # free up memory

img_size = len(x_train[0])
num_channels = len(x_train[0][0][0])
img_shape = x_train[0].shape #(img_size, img_size, num_channels)

# Determining augmentation
augment_data = False
if augment_data:
    if not os.path.isfile(more_data_file):
        print ("Processing augmentation...")
        more_images, more_labels = equalize_distribution(x_train, y_train, coords_train, sizes_train)
        if (len(more_images)>0):
            print('Saving data to pickle file...')
            try:
                with open(more_data_file, 'wb') as pfile:
                    pickle.dump(
                        {
                            FEATURES: more_images,
                            LABELS: more_labels,
                        },
                        pfile, pickle.HIGHEST_PROTOCOL)
                print ('Augmented data ({} images) cached in pickle file.'.format(len(more_images)))
            except Exception as e:
                print('Unable to save data to', more_data_file, ':', e)
                raise
    
    with open(more_data_file, 'rb') as f:
        more = pickle.load(f)
        x_more, y_more = more[FEATURES], more[LABELS]
        print ("Augmented data count:\t{}".format(len(x_more)))
        plot_images(x_more[0:9], cls_true=y_more[0:9])
        x_train = np.concatenate((x_train, x_more))
        y_train = np.concatenate((y_train, y_more))

# Free up some memory
del sizes_train, sizes_test, coords_train, coords_test

# Shuffle the training and testing data:
seq_train = np.arange(0, len(x_train))
seq_test = np.arange(0, len(x_test))
np.random.shuffle(seq_train)
np.random.shuffle(seq_test)
x_train = np.array([ x_train[i] for i in seq_train])
y_train = np.array([ y_train[i] for i in seq_train])
x_test = np.array([ x_test[i] for i in seq_test])
y_test = np.array([ y_test[i] for i in seq_test])

# TEMPORARY: Swap some of the test data into the training data and set the training data as the test data
# DO NOT USE the COORDS and SIZES arrays if this is being done.
#print("Doing test/train swapping...")
#z = len(x_test)
#x_tmp = x_test
#y_tmp = y_test
#x_test = np.array(x_train[0:z][:][:][:])
#y_test = np.array(y_train[0:z][:][:][:])
#x_train = np.append(x_train[z:][:][:][:], x_tmp, axis=0)
#y_train = np.append(y_train[z:][:][:][:], y_tmp, axis=0)

# Normalize the images (and save originals)
x_train_ = np.zeros_like(x_train, dtype=float32)
x_test_ = np.zeros_like(x_test, dtype=float32)
for i in range(0, len(x_train)):
    cv2.normalize(x_train[i], x_train_[i], 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
for i in range(0, len(x_test)):
    cv2.normalize(x_test[i], x_test_[i], 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Convert to the Dataset format
num_classes = len(np.unique(y_train))
y_train = dense_to_one_hot(y_train, num_classes) # One-hot encode all the labels
y_test = dense_to_one_hot(y_test, num_classes) # One-hot encode all the labels

traindata = DataSet(x_train_, y_train, reshape=False) # Use the normalized images (x_train_)
testdata = DataSet(x_test_, y_test, reshape=False) # Use the normalized images (x_test_)

data = Datasets(train=traindata, validation=None, test=testdata)
data.test.cls = np.argmax(data.test.labels, axis=1)
data.train.cls = np.argmax(data.train.labels, axis=1)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Shape:\t\t{}".format(img_shape))
print("- Num channels:\t{}".format(num_channels))
print("- Num classes:\t{}".format(num_classes))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
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

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global iteration_counter
    start_time = time.time()
    for i in range(iteration_counter, iteration_counter + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch, keep_prob: dropout}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
    iteration_counter += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Function for plotting examples of images from the test-set that have been mis-classified.
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = x_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


# ### Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images, y_true: labels, keep_prob: dropout}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# ### Helper-function for plotting the output of a convolutional layer
def plot_conv_layer(layer, image):
    feed_dict = {x: [image]}
    values = session.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# Helper-function for plotting an image.
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

### End of methods

#################################
#          PIPELINE             #
#################################

train_batch_size = 64
train_num_epochs = 300
test_batch_size = 256
iteration_counter = 0
learning_rate = 1e-3
dropout = 0.5 # % of outputs to keep

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

images = x_test[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Network architecture/layers
layer, _ = None, None
layer, _ = layer_conv1, weights_conv1 = new_conv_layer(input=x, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
layer, _ = layer_conv2, weights_conv2 = new_conv_layer(input=layer, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
layer, _ = layer_flat, num_features = flatten_layer(layer)
layer = layer_fc1 = new_fc_layer(input=layer, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
layer = layer_dropout = tf.nn.dropout(layer, keep_prob)
layer = layer_fc2 = new_fc_layer(input=layer, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)
y_pred = tf.nn.softmax(layer)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=0.5).minimize(cost)
#optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.0).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.initialize_all_variables())

for _ in range (0, 100):
    optimize(num_iterations=100)
    print_test_accuracy()
    
print_test_accuracy(show_example_errors=True)
#print_test_accuracy()
#optimize(num_iterations=1)
#print_test_accuracy()
#optimize(num_iterations=99) # We already performed 1 iteration above.
#print_test_accuracy(show_example_errors=False)
#optimize(num_iterations=900) # We performed 100 iterations above.
#print_test_accuracy(show_example_errors=False)
#optimize(num_iterations=9000) # We performed 1000 iterations above.
#print_test_accuracy(show_example_errors=False, show_confusion_matrix=False)

# Plot an image from the test-set which will be used as an example below.
#image1 = x_test[0]
#plot_image(image1)
#image2 = x_test[13]
#plot_image(image2)
#plot_conv_weights(weights=weights_conv1)
#plot_conv_layer(layer=layer_conv1, image=image1)
#plot_conv_layer(layer=layer_conv1, image=image2)
#plot_conv_weights(weights=weights_conv2, input_channel=0)
#plot_conv_weights(weights=weights_conv2, input_channel=1)
#plot_conv_layer(layer=layer_conv2, image=image1)
#plot_conv_layer(layer=layer_conv2, image=image2)
