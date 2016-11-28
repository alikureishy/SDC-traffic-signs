
from IPython.display import Image
Image('input_images/02_network_flowchart.png')
Image('input_images/02_convolution.png')


#!!get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 input_images 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 input_images 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.


# ## Load Data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
print (data)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))


data.test.cls = np.argmax(data.test.labels, axis=1)


# We know that MNIST input_images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of input_images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the input_images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


# ### Helper-function for plotting input_images
# Function used to plot 9 input_images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(input_images, actual_labels, label_predictor=None):
    assert len(input_images) == len(actual_labels) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(input_images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if label_predictor is None:
            xlabel = "True: {0}".format(actual_labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(actual_labels[i], label_predictor[i])

        # Show the classes as the label on the input_images-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Plot a few input_images to see if data is correct
# Get the first input_images from the test-set.
input_images = data.test.images[0:9]

# Get the true classes for those input_images.
cls_true = data.test.cls[0:9]

# Plot the input_images and labels using our helper-function above.
plot_images(input_images=input_images, actual_labels=cls_true)


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# ### Helper-function for creating a new Convolutional Layer

# This function creates a new convolutional layer in the computational graph for TensorFlow. Nothing is actually calculated here, we are just adding the mathematical formulas to the TensorFlow graph.
# 
# It is assumed that the input is a 4-dim tensor with the following dimensions:
# 
# 1. Image number.
# 2. Y-axis of each image.
# 3. X-axis of each image.
# 4. Channels of each image.
# 
# Note that the input channels may either be colour-channels, or it may be filter-channels if the input is produced from a previous convolutional layer.
# 
# The output is another 4-dim tensor with the following dimensions:
# 
# 1. Image number, same as input.
# 2. Y-axis of each image. If 2x2 pooling is used, then the height and width of the input input_images is divided by 2.
# 3. X-axis of each image. Ditto.
# 4. Channels produced by the convolutional filters.
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the input_images- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(input_images, 0) for each input pixel input_images.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(input_images)) == max_pool(relu(input_images)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


# ### Helper-function for flattening a layer
# 
# A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer.
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


# ### Helper-function for creating a new Fully-Connected Layer

# This function creates a new fully-connected layer in the computational graph for TensorFlow. Nothing is actually calculated here, we are just adding the mathematical formulas to the TensorFlow graph.
# 
# It is assumed that the input is a 2-dim tensor of shape `[num_images, num_inputs]`. The output is a 2-dim tensor of shape `[num_images, num_outputs]`.
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# ### Placeholder variables

# Placeholder variables serve as the input to the TensorFlow computational graph that we may change each time we execute the graph. We call this feeding the placeholder variables and it is demonstrated further below.
# 
# First we define the placeholder variable for the input input_images. This allows us to change the input_images that are input to the TensorFlow graph. This is a so-called tensor, which just means that it is a multi-dimensional vector or matrix. The data-type is set to `float32` and the shape is set to `[None, img_size_flat]`, where `None` means that the tensor may hold an arbitrary number of input_images with each image being a vector of length `img_size_flat`.
input_images = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='input_images')


# The convolutional layers expect `input_images` to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead `[num_images, img_height, img_width, num_channels]`. Note that `img_height == img_width == img_size` and `num_images` can be inferred automatically by using -1 for the size of the first dimension. So the reshape operation is:
x_image = tf.reshape(input_images, [-1, img_size, img_size, num_channels])


# Next we have the placeholder variable for the true labels associated with the input_images that were input in the placeholder variable `input_images`. The shape of this placeholder variable is `[None, num_classes]` which means it may hold an arbitrary number of labels and each label is a vector of length `num_classes` which is 10 in this case.
actual_hot_labels = tf.placeholder(tf.float32, shape=[None, 10], name='actual_hot_labels')


# We could also have a placeholder variable for the class-number, but we will instead calculate it using argmax. Note that this is a TensorFlow operator so nothing is calculated at this point.
actual_labels = tf.argmax(actual_hot_labels, dimension=1)


# ### Convolutional Layer 1
# 
# Create the first convolutional layer. It takes `x_image` as input and creates `num_filters1` different filters, each having width and height equal to `filter_size1`. Finally we wish to down-sample the image so it is half the size by using 2x2 max-pooling.
layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


# Check the shape of the tensor that will be output by the convolutional layer. It is (?, 14, 14, 16) which means that there is an arbitrary number of input_images (this is the ?), each image is 14 pixels wide and 14 pixels high, and there are 16 different channels, one channel for each of the filters.
layer_conv1


# ### Convolutional Layer 2
# 
# Create the second convolutional layer, which takes as input the output from the first convolutional layer. The number of input channels corresponds to the number of filters in the first convolutional layer.
layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)


# Check the shape of the tensor that will be output from this convolutional layer. The shape is (?, 7, 7, 36) where the ? again means that there is an arbitrary number of input_images, with each image having width and height of 7 pixels, and there are 36 channels, one for each filter.
layer_conv2


# ### Flatten Layer
# 
# The convolutional layers output 4-dim tensors. We now wish to use these as input in a fully-connected network, which requires for the tensors to be reshaped or flattened to 2-dim tensors.
layer_flat, num_features = flatten_layer(layer_conv2)

# Check that the tensors now have shape (?, 1764) which means there's an arbitrary number of input_images which have been flattened to vectors of length 1764 each. Note that 1764 = 7 input_images 7 input_images 36.
layer_flat
num_features


# ### Fully-Connected Layer 1
# 
# Add a fully-connected layer to the network. The input is the flattened layer from the previous convolution. The number of neurons or nodes in the fully-connected layer is `fc_size`. ReLU is used so we can learn non-linear relations.
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)


# Check that the output of the fully-connected layer is a tensor with shape (?, 128) where the ? means there is an arbitrary number of input_images and `fc_size` == 128.
layer_fc1


# ### Fully-Connected Layer 2
# 
# Add another fully-connected layer that outputs vectors of length 10 for determining which of the 10 classes the input image belongs to. Note that ReLU is not used in this layer.
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

layer_fc2


# ### Predicted Class

# The second fully-connected layer estimates how likely it is that the input image belongs to each of the 10 classes. However, these estimates are a bit rough and difficult to interpret because the numbers may be very small or large, so we want to normalize them so that each element is limited between zero and one and the 10 elements sum to one. This is calculated using the so-called softmax function and the result is stored in `hot_label_predictor`.
hot_label_predictor = tf.nn.softmax(layer_fc2)


# The class-number is the index of the largest element.
label_predictor = tf.argmax(hot_label_predictor, dimension=1)


# ### Cost-function to be optimized

# To make the model better at classifying the input input_images, we must somehow change the variables for all the network layers. To do this we first need to know how well the model currently performs by comparing the predicted output of the model `hot_label_predictor` to the desired output `actual_hot_labels`.
# 
# The cross-entropy is a performance measure used in classification. The cross-entropy is a continuous function that is always positive and if the predicted output of the model exactly matches the desired output then the cross-entropy equals zero. The goal of optimization is therefore to minimize the cross-entropy so it gets as close to zero as possible by changing the variables of the network layers.
# 
# TensorFlow has a built-in function for calculating the cross-entropy. Note that the function calculates the softmax internally so we must use the output of `layer_fc2` directly rather than `hot_label_predictor` which has already had the softmax applied.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=actual_hot_labels)


# We have now calculated the cross-entropy for each of the image classifications so we have a measure of how well the model performs on each image individually. But in order to use the cross-entropy to guide the optimization of the model's variables we need a single scalar value, so we simply take the average of the cross-entropy for all the image classifications.
cost = tf.reduce_mean(cross_entropy)


# ### Optimization Method

# Now that we have a cost measure that must be minimized, we can then create an optimizer. In this case it is the `AdamOptimizer` which is an advanced form of Gradient Descent.
# 
# Note that optimization is not performed at this point. In fact, nothing is calculated at all, we just add the optimizer-object to the TensorFlow graph for later execution.
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# ### Performance Measures

# We need a few more performance measures to display the progress to the user.
# 
# This is a vector of booleans whether the predicted class equals the true class of each image.
correct_prediction = tf.equal(label_predictor, actual_labels)


# This calculates the classification accuracy by first type-casting the vector of booleans to floats, so that False becomes 0 and True becomes 1, and then calculating the average of these numbers.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## TensorFlow Run

# ### Create TensorFlow session
# 
# Once the TensorFlow graph has been created, we have to create a TensorFlow session which is used to execute the graph.
session = tf.Session()


# ### Initialize variables
# 
# The variables for `weights` and `biases` must be initialized before we start optimizing them.
session.run(tf.initialize_all_variables())


# ### Helper-function to perform optimization iterations

# There are 55,000 input_images in the training-set. It takes a long time to calculate the gradient of the model using all these input_images. We therefore only use a small batch of input_images in each iteration of the optimizer.
# 
# If your computer crashes or becomes very slow because you run out of RAM, then you may try and lower this number, but you may then need to perform more optimization iterations.
train_batch_size = 64


# Function for performing a number of optimization iterations so as to gradually improve the variables of the network layers. In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples.  The progress is printed every 100 iterations.
# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of input_images and
        # y_true_batch are the true labels for those input_images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {input_images: x_batch,
                           actual_hot_labels: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# ### Helper-function to plot example errors

# Function for plotting examples of input_images from the test-set that have been mis-classified.
def plot_example_errors(cls_pred, gradings):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all input_images in the test-set.

    # gradings is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (gradings == False)
    
    # Get the input_images from the test-set that have been
    # incorrectly classified.
    input_images = data.test.images[incorrect]
    
    # Get the predicted classes for those input_images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those input_images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 input_images.
    plot_images(input_images=input_images[0:9],
                actual_labels=cls_true[0:9],
                predictions=cls_pred[0:9])


# ### Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all input_images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(actual_hot_labels=cls_true,
                          hot_label_predictor=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Helper-function for showing the performance

# Function for printing the classification accuracy on the test-set.
# 
# It takes a while to compute the classification for all the input_images in the test-set, that's why the results are re-used by calling the above functions directly from this function, so the classifications don't have to be recalculated by each function.
# 
# Note that this function can use a lot of computer memory, which is why the test-set is split into smaller batches. If you have little RAM in your computer and it crashes, then you can try and lower the batch-size.
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of input_images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the input_images from the test-set between index i and j.
        input_images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these input_images and labels.
        feed_dict = {input_images: input_images,
                     actual_hot_labels: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(label_predictor, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified input_images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # input_images divided by the total number of input_images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, gradings=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# ## Performance before any optimization
# 
# The accuracy on the test-set is very low because the model variables have only been initialized and not optimized at all, so it just classifies the input_images randomly.
print_test_accuracy()


# ## Performance after 1 optimization iteration
# 
# The classification accuracy does not improve much from just 1 optimization iteration, because the learning-rate for the optimizer is set very low.
optimize(num_iterations=1)

print_test_accuracy()


# ## Performance after 100 optimization iterations
# 
# After 100 optimization iterations, the model has significantly improved its classification accuracy.
optimize(num_iterations=99) # We already performed 1 iteration above.

print_test_accuracy(show_example_errors=True)


# ## Performance after 1000 optimization iterations
# 
# After 1000 optimization iterations, the model has greatly increased its accuracy on the test-set to more than 90%.
optimize(num_iterations=900) # We performed 100 iterations above.

print_test_accuracy(show_example_errors=True)


# ## Performance after 10,000 optimization iterations
# 
# After 10,000 optimization iterations, the model has a classification accuracy on the test-set of about 99%.
optimize(num_iterations=9000) # We performed 1000 iterations above.

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


# ## Visualization of Weights and Layers
# 
# In trying to understand why the convolutional neural network can recognize handwritten digits, we will now visualize the weights of the convolutional filters and the resulting output input_images.

# ### Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the input_images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Helper-function for plotting the output of a convolutional layer
def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed actual_hot_labels because it is
    # not used in this calculation.
    feed_dict = {input_images: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output input_images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the input_images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Input Images

# Helper-function for plotting an image.
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()


# Plot an image from the test-set which will be used as an example below.
image1 = data.test.images[0]
plot_image(image1)


# Plot another example image from the test-set.
image2 = data.test.images[13]
plot_image(image2)


# ### Convolution Layer 1

# Now plot the filter-weights for the first convolutional layer.
# 
# Note that positive weights are red and negative weights are blue.
plot_conv_weights(weights=weights_conv1)


# Applying each of these convolutional filters to the first input image gives the following output input_images, which are then used as input to the second convolutional layer. Note that these input_images are down-sampled to 14 input_images 14 pixels which is half the resolution of the original input image.
plot_conv_layer(layer=layer_conv1, image=image1)


# The following input_images are the results of applying the convolutional filters to the second image.
plot_conv_layer(layer=layer_conv1, image=image2)


# It is difficult to see from these input_images what the purpose of the convolutional filters might be. It appears that they have merely created several variations of the input image, as if light was shining from different angles and casting shadows in the image.

# ### Convolution Layer 2

# Now plot the filter-weights for the second convolutional layer.
# 
# There are 16 output channels from the first conv-layer, which means there are 16 input channels to the second conv-layer. The second conv-layer has a set of filter-weights for each of its input channels. We start by plotting the filter-weigths for the first channel.
# 
# Note again that positive weights are red and negative weights are blue.
plot_conv_weights(weights=weights_conv2, input_channel=0)


# There are 16 input channels to the second convolutional layer, so we can make another 15 plots of filter-weights like this. We just make one more with the filter-weights for the second channel. 
plot_conv_weights(weights=weights_conv2, input_channel=1)


# It can be difficult to understand and keep track of how these filters are applied because of the high dimensionality.
# 
# Applying these convolutional filters to the input_images that were ouput from the first conv-layer gives the following input_images.
# 
# Note that these are down-sampled yet again to 7 input_images 7 pixels which is half the resolution of the input_images from the first conv-layer.
plot_conv_layer(layer=layer_conv2, image=image1)


# And these are the results of applying the filter-weights to the second image.
plot_conv_layer(layer=layer_conv2, image=image2)
