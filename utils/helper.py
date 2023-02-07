from config import *
from collections import defaultdict
import os
import re
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import time


class Helper:        
    def normalize_caption(self, input_str):
        input_str = tf.strings.lower(input_str)
        
        # Remove special characters and numbers
        input_str = tf.strings.regex_replace(input_str, pattern='[^a-z\s]+', rewrite='')
        
        # Strip leading and trailing whitespaces
        input_str = tf.strings.strip(input_str)
        
        # Replace tabs with spaces
        input_str = tf.strings.regex_replace(input_str, pattern='\t', rewrite=' ')
        
        return input_str

    def load_train_data_to_tensor(self, caption_file=None, train_file=None, image_dir=None, cnn_model = None, tokenizer=None):
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
                standardize=self.normalize_caption,
                output_sequence_length=config['max_length'])

            tokenizer.adapt(caption_dataset)

        caption_vectors = caption_dataset.map(lambda x: tokenizer(x))

        # word_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
        # index_to_word = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)

        # Get unique images
        unique_image_paths = sorted(set(all_image_paths))
        image_dataset = tf.data.Dataset.from_tensor_slices(unique_image_paths)
        image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

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
                            self.map_func, [item1, item2], [tf.float32, tf.int64]
                      )
                      , num_parallel_calls=tf.data.AUTOTUNE)
        # Shuffle and batch
        dataset = dataset.shuffle(config['buffer_size']).batch(config['batch_size'])
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset    
    # store the features to a numpy file
    def load_image(self, image_path, size=(260, 260), preprocessor=tf.keras.applications.EfficientNetB0):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)  # EfficientNetB2 expects this input shape
        img = preprocessor(img)
        return img, image_path
    
    def map_func(self, image_path, caption):
        path = image_path.decode('utf-8') + '.npy'
        img_tensor = np.load(path)
        return img_tensor, caption

def create_look_ahead_mask(sequence_length):
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask 

def create_padding_mask(decoder_token_ids):
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
  
    return seq[:, tf.newaxis, :]

def get_angles(pos, i, d_model):
    '''
      Notice that the equations of positional encoding above is about 2i, and this
      function is about i, so that we compute (i // 2).

      "Angle" means the expression inside of sin and cosine function.
    '''
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding_1d(position, D):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  # column vector
                            np.arange(D)[np.newaxis, :],  # row vector
                            D)

    # Apply the sine function to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply the cosine function to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def positional_encoding_2d(row, col, D):
    assert D % 2 == 0
    # first D/2 encode row embedding and second D/2 encode column embedding
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)

    angle_rads_row = get_angles(row_pos,
                                np.arange(D // 2)[np.newaxis, :],
                                D // 2)
    angle_rads_col = get_angles(col_pos,
                                np.arange(D // 2)[np.newaxis, :],
                                D // 2)
    
    # apply sin and cos to odd and even indices resp.
    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])

    pos_encoding = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def point_wise_feed_forward_network(
    emb_dim,  # Input/output dimensionality (or Embedding dim)
    fc_dim  # Inner-layer dimensionality (or FC dim)
):

    return tf.keras.Sequential([
        # Shape `(batch_size, seq_len, fc_dim)`.
        tf.keras.layers.Dense(fc_dim, activation='relu'),
        # Shape `(batch_size, seq_len, emb_dim)`.
        tf.keras.layers.Dense(emb_dim)
    ])

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, emb_dim, num_heads, fc_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderBlock, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                      key_dim=emb_dim,
                                                      dropout=dropout_rate)

        self.ffn = point_wise_feed_forward_network(emb_dim=emb_dim,
                                                   fc_dim=fc_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
            `mask` is padding mask
        """
        attn_output = self.mha(query=x, key=x, value=x,
                               training=training, attention_mask=mask)

        # (batch_size, input_seq_len, fully_connected_dim)
        out1 = self.layernorm1(attn_output + x, training=training)

        # (batch_size, input_seq_len, fully_connected_dim)
        ffn_output = self.ffn(out1, training=training)

        ffn_output = self.dropout1(ffn_output, training=training)

        # (batch_size, input_seq_len, fully_connected_dim)
        encoder_layer_out = self.layernorm2(ffn_output + out1, training=training)

        return encoder_layer_out

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 num_layers,
                 emb_dim,  # Input/output dimensionality (or Embedding dim).
                 num_heads,
                 fc_dim,  # Inner-layer dimensionality (or FC dim).
                 row_size, col_size,    # Shape of grid features
                 dropout_rate=0.1,
                 layernorm_eps=1e-6):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_layers = num_layers

        # Embeddings (it's just a Dense layer)
        self.embedding = tf.keras.layers.Dense(emb_dim, activation='relu')
        # Positional encoding 2D
        self.pos_encoding = positional_encoding_2d(row_size, col_size, emb_dim)

        # Encoder layers.
        self.enc_layers = [EncoderBlock(emb_dim=emb_dim,
                                        num_heads=num_heads,
                                        fc_dim=fc_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps)
            for _ in range(num_layers)]

        # Dropout.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        # Sum up embeddings and positional encoding.
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]

        # Add dropout.
        x = self.dropout(x, training=training)

        # N encoder blocks.
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # Shape `(batch_size, input_seq_len, emb_dim)

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 emb_dim,  # Input/output dimensionality (or Embedding dim).
                 num_heads,
                 fc_dim,  # Inner-layer dimensionality (or FC dim).
                 dropout_rate=0.1,
                 layernorm_eps=1e-6):
        super().__init__()

        # Masked multi-head self-attention.
        self.mha_masked = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            # Size of each attention head for query Q and key K.
            key_dim=emb_dim,
            dropout=dropout_rate
        )
        # Multi-head cross-attention.
        self.mha_cross = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            # Size of each attention head for query Q and key K.
            key_dim=emb_dim,
            dropout=dropout_rate
        )

        # Point-wise feed-forward network.
        self.ffn = point_wise_feed_forward_network(emb_dim, fc_dim)

        # Layer normalization.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # The encoder output shape is `(batch_size, input_seq_len, emb_dim)`.

        attn_masked, attn_weights_masked = self.mha_masked(
            query=x,
            value=x,
            key=x,
            # A boolean mask that prevents attention to certain positions.
            attention_mask=look_ahead_mask,
            # Shape `(batch_size, target_seq_len, emb_dim)`.
            return_attention_scores=True,
            training=training
        )

        out1 = self.layernorm1(attn_masked + x)

        attn_cross, attn_weights_cross = self.mha_cross(
            query=out1,
            value=enc_output,
            key=enc_output,
            # A boolean mask that prevents attention to certain positions.
            attention_mask=padding_mask,
            # Shape `(batch_size, target_seq_len, emb_dim)`.
            return_attention_scores=True,
            training=training
        )

        out2 = self.layernorm2(attn_cross + out1)

        # Shape `(batch_size, target_seq_len, emb_dim)`.
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout1(ffn_output, training=training)
        
        # Shape `(batch_size, target_seq_len, emb_dim)`.
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_masked, attn_weights_cross

