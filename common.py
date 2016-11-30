import numpy as np
from collections import namedtuple

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

