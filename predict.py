
import os
import urllib
import json

import numpy as np
import matplotlib.pyplot as plt

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf

# from Colab: TensorFlow with GPU -- https://colab.research.google.com/notebooks/gpu.ipynb
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

#import tensorflow as tf
import tensorflow_hub as hub
#import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds

from tensorflow.keras import layers

print(tf.version)

#!nvidia-smi
# import warnings
# warnings.filterwarnings('ignore')

# from https://www.tensorflow.org/datasets/overview
# pip install -q tfds-nightly tensorflow matplotlib
# for local setup


# Load the dataset with TensorFlow Datasets.
dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
print(dataset_info)

# from TF example: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c03_exercise_flowers_with_transfer_learning_solution.ipynb#scrollTo=oXiJjX0jfx1o
# (training_set, validation_set), dataset_info = tfds.load(
#     'oxford_flowers102',
#     split=['train[:70%]', 'train[70%:]'],
#     with_info=True,
#     as_supervised=True,
# )


# Create a training set, a validation set and a test set.
test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

print(training_set)

one_im = training_set.take(1)  # Only take a single example

print(one_im)



# Get the number of examples in each set from the dataset info.
num_training_examples = 0
num_validation_examples = 0
num_test_examples = 0

for example in training_set:
  num_training_examples += 1

for example in validation_set:
  num_validation_examples += 1

for example in test_set:
  num_test_examples += 1

print(f'Total Number of Training Images: {num_training_examples}')
print(f'Total Number of Validation Images: {num_validation_examples}')
print(f'Total Number of Test Images: {num_test_examples}')
print('\n\n')


# Get the number of classes in the dataset from the dataset info.

num_classes = dataset_info.features['label'].num_classes
print(f'Total Number of Classes: {num_classes}')

# Print the shape and corresponding label of 3 images in the training set.

for i, example in enumerate(training_set.take(3)):
  print(f'Image {i+1} shape: {example[0].shape} label: {example[1]}')

training_set.take(1)

# Plot 1 image from the training set. Set the title 
# of the plot to the corresponding image label.


# see https://www.tensorflow.org/guide/data

#(image, label) = training_set.take(1)
for image, label in training_set.take(1):
  print(image.shape, label)
image = image.numpy()

plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.title(f'{label}')
plt.colorbar()
#plt.grid(False)
plt.show()

"""## Label Mapping

You'll also need to load in a mapping from label to category name. You can find this in the file label_map.json. It's a JSON object which you can read in with the [json module](https://docs.python.org/3.7/library/json.html). This will give you a dictionary mapping the integer coded labels to the actual names of the flowers.
"""

# import json
#import pandas as pd 

labels_url= "https://github.com/udacity/intro-to-ml-tensorflow/blob/master/projects/p2_image_classifier/label_map.json"
with open('label_map.json', 'r') as f:
    class_names = json.load(f)
#class_names = pd.read_json(labels_url, orient='records', dtype='dict')

#print(class_names[str(label.numpy())])


plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.title(class_names[str(label.numpy())])
plt.colorbar()
plt.grid(False)
plt.show()


## Create Pipeline

# Create a pipeline for each set.
IMAGE_RES = 224

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches = training_set.cache().shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)



# Build and train your network.
# from tensorflow.keras import layers

# Create a Feature Extractor
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES, 3))
# Freeze the Pre-Trained Model
feature_extractor.trainable = False
# Attach a classification head
model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(num_classes, activation='softmax')
])

print( model.summary())

print('GPU Available: ', tf.test.is_gpu_available())

model.compile(
  optimizer= adam',
  loss= 'sparse_categorical_crossentropy',
  metrics= ['accuracy'])

EPOCHS = 20

# Stop training when there is no improvement in the validation loss for 5 consecutive epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= 3)

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    callbacks=[early_stopping])



"""## References

https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c03_exercise_flowers_with_transfer_learning_solution.ipynb

"""