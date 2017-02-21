
import cv2
import numpy as np
from common import *
import os
import matplotlib.image as mpimg
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot
from numpy import float32

# Some image manipulation routines
# Needed for subsequent processing

# Crops an image based on the coordinates obtained from the input data
# Cropping of data is not useful in this project. But I am keeping it here,
# in case it is needed in the future.
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

# Placeholder, in case more image transformations are needed in the future
def sharpen_blur(image):
    return image

# Placeholder, in case more image transformations are needed in the future
def rotate(image):
    return image

# Placeholder, in case more image transformations are needed in the future
def brighten_darken(image):
    return image

# Placeholder, in case more image transformations are needed in the future
def push_away(image, newshape):
    return image

# Placeholder, in case more image transformations are needed in the future
def change_perspective(image):
    return image

# Placeholder, in case more image transformations are needed in the future
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
    dim = (int(np.ceil(image.shape[0] * r_x)), int(np.ceil(image.shape[1] * r_y)))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

# This is the actual data generation routine. It's aim is to equalize the distribution of training samples
# so that training is not only more generalized, but also does not skew towards specific classes.
# This method also prints out the distribution of examples across the training classes.
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
                    
                    # TODO: Move transformation code over to another location in the source
                    
                    more_images.append(image)
                    more_labels.append(cls)
        
    return more_images, more_labels


def print_distribution(xs, ys):
    num_images = len(xs)
    (unique, counts) = np.unique(ys, return_counts=True)
    descending = np.argsort(counts)[::-1] # Sort the classes in descending order based on 'counts' array values
    for cls in descending:
        print ("Class\t{}:\t{} examples \t[% = {}]".format(cls, counts[cls], counts[cls]*100/num_images))

# Shuffle the training and testing data:
def shuffle (data):
    seq = np.arange(0, len(data.images))
    np.random.shuffle(seq)
    images = np.array([ data.images[i] for i in seq])
    pre_images = np.array([ data.pre_images[i] for i in seq])
    labels = np.array([ data.labels[i] for i in seq])
    hot_labels = np.array([ data.hot_labels[i] for i in seq])
    return Data(images=images, pre_images=pre_images, labels=labels, hot_labels=hot_labels, count=len(images), batch_size=data.batch_size)

def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f)
    return files

def read_images(dir, mappings, num_classes):
    pre_images = np.zeros((1000, 32, 32, 3))
    labels = []
    names = []
    
    files = listdir_nohidden(dir)
    for i in range(len(files)):
        file = files[i]
        if file in mappings:
            tmp = mpimg.imread(os.path.join(dir, file))
            pre_image = resize(tmp, 32)
            label = mappings[file]
            
            pre_images[i] = pre_image
            labels.append(label)
            names.append(file)
        else:
            print (file, " not in mappings")
    
    pre_images = pre_images[0:len(labels)]
    labels = np.asarray(labels)
#    print ("Labels: ", labels.shape)
#    print ("Pre_images: ", pre_images.shape)
    labels_onehot = dense_to_one_hot(labels, 43)
#    print ("Onehot: ", labels_onehot.shape)
    images = np.zeros_like(pre_images, dtype=float32)
#    print ("Images: ", images.shape)
    for i in range(0, len(pre_images)):
        cv2.normalize(pre_images[i], images[i], 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return pre_images, images, labels, labels_onehot, names