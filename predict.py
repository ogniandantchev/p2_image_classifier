import numpy as np
import matplotlib.pyplot as plt
import json

import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from PIL import Image

IMAGE_RES= 224

print()
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
# `tf.config.list_physical_devices('GPU')` instead.
print(tf.config.list_physical_devices('GPU'))
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# def process_image(image):
#     image = tf.cast(image, tf.float32)
#     # you can also do this for conversion tf.image.convert_image_dtype(x, dtype=tf.float16, saturate=False)
#     image = tf.image.resize(image, (image_size, image_size)).numpy()
#     image /= 255
#     return image

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image

def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)
    
    pred_image = model.predict(expanded_test_image)
    values, indices = tf.math.top_k(pred_image, k=top_k)
    probs = values.numpy()[0]
    classes = indices.numpy()[0] + 1
    
    # preapere the result for presenting
    probs = list(probs)
    classes = list(map(str, classes))
    
    return probs, classes

class_names = []

if __name__ =='__main__':

#Initailize the parser
    parser = argparse.ArgumentParser(description='Flowers-102 Image Classifier')
#Add Args
    parser.add_argument('image_path',  action= 'store', help="Image path/file")
    parser.add_argument('model', type= str, help= "Trained model file in ")
    parser.add_argument('--top_k', type= int, help= 'Top K most likely classes to print')
    parser.add_argument('--category_names', type= str, help= 'File name of category labels, JSON format')
    
#Parse Arguments
    args = parser.parse_args()

    image_path = args.image_path

    model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())

    if args.top_k is None and args.category_names is None:
        probs, classes = predict(image_path, model)
        print("Probabilities and classes for the image: ")
    elif args.top_k is not None:
        top_k = int(args.top_k)
        probs, classes = predict(image_path, model, top_k)
        print(f"The top {top_k} probabilities and classes for the image: ")
    elif args.category_names is not None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        probs, classes = predict(image_path, model)
        print("Probabilities and classes for the image: ")
        classes = [class_names[c] for c in  classes]
            
    for prob, c in zip(probs, classes):
        print(f'{c}:  {prob:.2%}')
    
    print(f'The flower is: {classes[0]}, most probably.')

    # class_names[str(value)]