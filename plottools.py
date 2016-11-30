import matplotlib
import matplotlib.pyplot as plt

# Function used to plot 9 input_images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(images, actual_labels, predictions=None):
    assert len(images) <= len(actual_labels) <= 9
    img_shape = images[0].shape
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if predictions is None:
            xlabel = "True: {0}".format(actual_labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(actual_labels[i], predictions[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# Function for plotting examples of input_images from the test-set that have been mis-classified.
def plot_example_errors(data, predictions, gradings):
    incorrect = (gradings == False)
    images = data.images[incorrect] #images = [ data.images[i] for i in range(data.count) where incorrect[i]]
    predictions = predictions[incorrect]
    actuals = data.labels[incorrect]
    plot_images(images=images[0:9], actual_labels=actuals[0:9], predictions=predictions[0:9])

# Helper-function for plotting an image.
def plot_image(image):
    img_shape = image.shape
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    plt.show()
