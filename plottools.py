import matplotlib.pyplot as plt
import numpy as np

# Function used to plot 9 input_images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(images, actual_labels, predictions=None, titles=None):
    img_shape = images[0].shape
    h=int(round(np.sqrt(len(images))))
    v=int(np.ceil(len(images)/h))
#     diff = (h*v) - len(images)
#     sample = np.zeros((20, 20, 3), dtype=np.uint8)
#     for _ in range(diff):
#         sections[0].append(Image("--Blank--", sample, None))
#     sections = np.reshape(np.array(sections), (v,h))

    fig, axes = plt.subplots(v, h,figsize=(11,11))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape(img_shape), cmap='binary')
            if predictions is None:
                xlabel = "True: {0}".format(actual_labels[i])
            else:
                xlabel = "True: {0}, Predicted: {1}".format(actual_labels[i], predictions[i])
            if titles is not None:
                ax.title(titles[i])
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.imshow(np.zeros(img_shape), cmap='binary')
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.show()

# Function for plotting examples of input_images from the test-set that have been mis-classified.
def plot_example_errors(data, predictions, gradings):
    incorrect = [i for (i,grade) in enumerate(gradings) if grade == False]
    images = data.images[incorrect] #images = [ data.images[i] for i in range(data.count) where incorrect[i]]
    predictions = predictions[incorrect]
    actuals = data.labels[incorrect]
    plot_images(images=images, actual_labels=actuals, predictions=predictions)

# Helper-function for plotting an image.
def plot_image(image):
    img_shape = image.shape
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    plt.show()
