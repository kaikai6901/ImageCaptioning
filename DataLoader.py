import os
from collections import defaultdict
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pickle

start_seq = 'start'
end_seq = 'end'

image_dir = './Images'
captions_path = './captions.txt'
data_dir = './data'

train_image_path = './data/train_image.txt'
test_image_path = './data/test_image.txt'

max_length = 34
vocab_length = 2019
batch_size = 64
# Change image size for pre-trained model
img_size = (300, 300)

def get_vectorization():
    def normalize_caption(input_str):
        input_str = tf.strings.lower(input_str)

        # Remove special characters and numbers
        input_str = tf.strings.regex_replace(
            input_str, pattern='[^a-z\s]+', rewrite='')

        # Strip leading and trailing whitespaces
        input_str = tf.strings.strip(input_str)

        # Replace tabs with spaces
        input_str = tf.strings.regex_replace(input_str, pattern='\t', rewrite=' ')

        return input_str
    list_caps = []
    with open(captions_path) as file:
        lines = file.readlines()
        for line in lines[1:]:
            data = line.split(',')
            cap = start_seq + ' ' + ' '.join(data[1:]).strip() + ' ' + end_seq
            list_caps.append(cap)
    cap_tf_dataset = tf.data.Dataset.from_tensor_slices(list_caps)
    vectorization = tf.keras.layers.TextVectorization(
        max_tokens=vocab_length,
        standardize=normalize_caption,
        output_sequence_length=max_length)
    vectorization.adapt(cap_tf_dataset)
    return vectorization

def get_text_transfrom():
    with open(os.path.join(data_dir, 'index_to_word.pickle'), 'rb') as handle:
        index_to_word = pickle.load(handle)
    
    with open(os.path.join(data_dir, 'word_to_index.pickle'), 'rb') as handle:
        word_to_index = pickle.load(handle)
    
    return word_to_index, index_to_word

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, image_path

def get_extraction_model():
    image_model = tf.keras.applications.EfficientNetB3(
    include_top=False, weights='imagenet')
    img_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    extraction_model = tf.compat.v1.keras.Model(
        img_input, hidden_layer)
    return extraction_model

