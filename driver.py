
#!!get_ipython().magic('matplotlib inline')
import os
import cv2
import pickle
import numpy as np
import matplotlib
from plottools import plot_example_errors
matplotlib.use('TkAgg')
from numpy import float32
import tensorflow as tf
from datatools import *
from common import *
from traintools import *
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot

# ## Load Data
#from tensorflow.examples.tutorials.mnist import input_data
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

train = Data(x_train_norm, x_train, y_train, y_train_hot, len(x_train), batch_size=128) # Use the normalized input_images (x_train_norm)
test = Data(x_test_norm, x_test, y_test, y_test_hot, len(x_test), batch_size=256) # Use the normalized input_images (x_test_norm)
meta = Meta(x_test_norm[0].shape, len(x_test[0][0][0]), num_classes)

print("Size of:")
print("- Training-set:\t\t{}".format(train.count))
print("- Test-set:\t\t{}".format(test.count))
print("- Shape:\t\t{}".format(meta.image_shape))
print("- Num channels:\t{}".format(meta.num_channels))
print("- Num classes:\t{}".format(meta.num_classes))

#################################
#          PIPELINE             #
#################################

params = Params (\
                 num_train_epochs=0,
                 learning_rate=1e-3,
                 dropout=0.5, # this is actually keep_prob
                 validation_set_size=int(0.10 * train.count),
                 validation_frequency=100, # Validation is performed every 'n' batches
                 training_accuracy_threshold=0.92, \
                 do_checkpointing=False)

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
actual_labels = tf.argmax(actual_hot_labels, dimension=1, name="actual_labels")
keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)

# Network architecture/layers
layer, _ = None, None
layer, _ = layer_conv1, weights_conv1 = new_conv_layer(input=images, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True, name='layer_conv1')
#layer = layer_dropout1 = tf.nn.dropout(layer, keep_prob, name='layer_dropout1')

layer, _ = layer_conv2, weights_conv2 = new_conv_layer(input=layer, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True, name='layer_conv2')
#layer = layer_dropout2 = tf.nn.dropout(layer, keep_prob, name='layer_dropout2')

layer, _ = layer_conv3, weights_conv3 = new_conv_layer(input=layer, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3, use_pooling=True, name='layer_conv3')
#layer = layer_dropout3 = tf.nn.dropout(layer, keep_prob, name='layer_dropout3')

layer, _ = layer_flat1, num_features = flatten_layer(layer, name='layer_flat1')

layer = layer_fc1 = new_fc_layer(input=layer, num_inputs=num_features, num_outputs=fc_size, use_relu=True, name='layer_fc1')
layer = layer_dropout4 = tf.nn.dropout(layer, keep_prob, name='layer_dropout4')

layer = logits = layer_fc2 = new_fc_layer(input=layer, num_inputs=fc_size, num_outputs=num_classes, use_relu=False, name='layer_fc2')
#layer = logits = layer_dropout5 = tf.nn.dropout(layer, keep_prob, name='layer_dropout5')

hot_label_predictor = tf.nn.softmax(logits, name='hot_label_predictor')
label_predictor = tf.argmax(hot_label_predictor, dimension=1, name='label_predictor')
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actual_hot_labels, name='cross_entropy')
cost = tf.reduce_mean(cross_entropy, name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate, name='optimizer').minimize(cost)
#correct_prediction = tf.equal(label_predictor, actual_labels,name='correct_prediction')
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

#
# Finally, we're down to the point when the network is trained
# and the test set is run against it periodically to determine accuracy.
# The parameters used here are to be defined earlier on in this notebook.
#
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
# If a checkpoint has been saved already (checkpoints/checkpoint.chk) this will load
# and continue training where it last was.
# If the params.num_train_epochs is set to None, it will train indefinitely, until 
# params.training_accuracy_threshold is attained.
# If params.do_checkpointing is set to True, the checkpoint will keep getting updated
# with the latest weights/biases etc. For the time being, it is said to False, to
# retain the original trained settings.
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
        train_epoch(optimizer, label_predictor, train, meta, params, session, images, actual_hot_labels, keep_prob)
    
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
    correct, accuracy, predictions, gradings = check_accuracy(label_predictor, test, meta, params, session, images, actual_hot_labels, keep_prob)
    print("Final accuracy on Test-Set: {0:.1%} ({1} / {2})".format(accuracy, correct, test.count))

print ("\nDONE!")
#plot_example_errors(test, predictions, gradings)

#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=0.5).minimize(cost)
#optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.0).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost)

# Reading in input data for testing real images
real_dir = "real_data"
mappings = {"130_km_1.jpg": 8, # This is incorrect, but the closest match
            "130_km_2.jpg": 8, # Same as above
            "30_km_1.jpg": 1,
            "50_km_1.jpg": 2,
            "50_km_2.jpg": 2,
            "50_km_3.jpg": 2,
            "80_km_1.jpg": 5,
            "60_km_1.jpg": 3,
            "slippery_1.jpg": 23,
            "slippery_2.jpg": 23,
            "children_1.jpg": 28,
            "children_2.jpg": 28,
            "children_3.jpg": 28,
            "children_4.jpg": 28,
            "roundabout_1.jpg": 40,
            "bicycle_1.jpg": 29,
            "bicycle_2.jpg": 29,
            "no_entry_1.jpg": 17}

preimages, imgs, labels, labelsonehot = read_images(real_dir, mappings, 43)
data = Data(pre_images = preimages, images=imgs, labels=labels, hot_labels=labelsonehot, count=len(preimages), batch_size=100)
data = shuffle(data)
correct, acc, predictions, gradings = check_accuracy(label_predictor, data, meta, params, session, images, actual_hot_labels, keep_prob)
print ("Real data results:")
print ("\t#Samples: {}".format(len(imgs)))
print ("\tAccuracy: {:.1%}".format(acc))
print ("Here are some sample errors:")
plot_example_errors(data, predictions, gradings)
