from config import *
from collections import defaultdict
import os
import re
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
@tf.function
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=(224, 224))  # EfficientNetB0 expects this input shape
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, image_path

@tf.function
def map_func(image_path, caption):
    path = image_path.decode('utf-8') + '.npy'
    img_tensor = np.load(path)
    return img_tensor, caption

@tf.function
def normalize_caption(input_str):
    input_str = tf.strings.lower(input_str)
    
    # Remove special characters and numbers
    input_str = tf.strings.regex_replace(input_str, pattern='[^a-z\s]+', rewrite='')
    
    # Strip leading and trailing whitespaces
    input_str = tf.strings.strip(input_str)
    
    # Replace tabs with spaces
    input_str = tf.strings.regex_replace(input_str, pattern='\t', rewrite=' ')
    
    return input_str
class Helper:        
    def load_train_data_to_tensor(self, caption_file=None, train_file=None, image_dir=None, cnn_model = None, tokenizer=None):
        if image_dir == None:
            image_dir = config['image_dir']

        captions_map = defaultdict(list)
        if caption_file == None:
            caption_file = config['caption_file']

        with open(caption_file) as file:
            lines = file.readlines()
            for line in lines[1:]:
                data = line.split(',')
                image_id = data[0]
                caption = config['start_seq'] + ' ' + ' '.join(data[1:]).strip() + ' ' + config['end_seq']
                captions_map[image_id].append(caption)
        if train_file == None:
            train_file = config['train_file']

        train_image = []
        with open(train_file) as file:
            lines = file.readlines()
            train_image = [line.strip() for line in lines]
        all_captions = []
        all_image_paths = []
        for image_id in train_image:
            all_captions.extend(captions_map[image_id])
            all_image_paths.extend([os.path.join(image_dir, image_id)] * len(captions_map[image_id]))

        caption_dataset = tf.data.Dataset.from_tensor_slices(all_captions)

        # Parameters for tokenizer

        # Tokenizer
        if tokenizer == None:
            tokenizer = tf.keras.layers.TextVectorization(
                max_tokens=config['vocab_size'],
                standardize=normalize_caption,
                output_sequence_length=config['max_length'])

            tokenizer.adapt(caption_dataset)

        caption_vectors = caption_dataset.map(lambda x: tokenizer(x))

        # word_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
        # index_to_word = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)

        # Get unique images
        unique_image_paths = sorted(set(all_image_paths))
        image_dataset = tf.data.Dataset.from_tensor_slices(unique_image_paths)
        image_dataset = image_dataset.map(tf.function(load_image), num_parallel_calls=tf.data.AUTOTUNE).batch(16)

        for image, path in tqdm(image_dataset):
            # batch_features shape == (16, 8, 8, 1408) (16 is batch size)
            batch_features = cnn_model(image)
            # after reshaping, batch_features shape == (16, 64, 1408)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                path_of_feature = path_of_feature[18:]
                np.save(path_of_feature, bf.numpy())

        dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, caption_vectors))
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
                            map_func, [item1, item2], [tf.float32, tf.int64]
                      )
                      , num_parallel_calls=tf.data.AUTOTUNE)
        # Shuffle and batch
        dataset = dataset.shuffle(config['buffer_size']).batch(config['batch_size'])
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset, tokenizer   
    # store the features to a numpy file

def create_look_ahead_mask(sequence_length):
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask 

def create_padding_mask(decoder_token_ids):
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
  
    return seq[:, tf.newaxis, :]


