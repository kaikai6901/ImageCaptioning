import os
from collections import defaultdict
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pickle
from DataLoader import *

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def make_padding_mask(decoder_token_ids):
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    return seq[:, tf.newaxis, :]

def make_look_ahead_mask(sequence_length):
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask 

def get_angles(pos, i, dim_model):
    angle = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim_model))
    return pos * angle

def pos_encode_1D(position, D):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  
                            np.arange(D)[np.newaxis, :],  
                            D)

    # Apply the sine function to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply the cosine function to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def pos_encode_2D(row, col, D):
    assert D % 2 == 0
    # first D/2 encode row embedding and second D/2 encode column embedding
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)

    angle_row = get_angles(row_pos,
                                np.arange(D // 2)[np.newaxis, :],
                                D // 2)
    angle_col = get_angles(col_pos,
                                np.arange(D // 2)[np.newaxis, :],
                                D // 2)
    
    # apply sin and cos to odd and even indices resp.
    angle_row[:, 0::2] = np.sin(angle_row[:, 0::2])
    angle_row[:, 1::2] = np.cos(angle_row[:, 1::2])
    angle_col[:, 0::2] = np.sin(angle_col[:, 0::2])
    angle_col[:, 1::2] = np.cos(angle_col[:, 1::2])

    pos_encoding = np.concatenate([angle_row, angle_col], axis=1)[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def point_wise_ffn(
    embedded_dim,  # Input/output dimensionality (or Embedding dim)
    fc_dims  # Inner-layer dimensionality (or FC dim)
):

    return tf.keras.Sequential([
        # Shape `(batch_size, seq_len, fc_dim)`.
        tf.keras.layers.Dense(fc_dims, activation='relu'),
        # Shape `(batch_size, seq_len, emb_dim)`.
        tf.keras.layers.Dense(embedded_dim)
    ])

class CustomEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embedded_dim, number_heads, fc_dims, drop_rate=0.1, layernorm_eps=1e-6):
        super(CustomEncoderBlock, self).__init__()
        self.MHA = tf.keras.layers.MultiHeadAttention(num_heads=number_heads, key_dim=embedded_dim, dropout=drop_rate)

        self.FeedForward = point_wise_ffn(embedded_dim=embedded_dim, fc_dims=fc_dims)

        self.CustomLayerNorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.CustomLayerNorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.CustomDropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, data, is_training, padding_mask):
        att = self.MHA(
            query=data, key=data, value=data,
            training = is_training, ##  in training mode (adding dropout) or in inference mode (no dropout)
            attention_mask = padding_mask
        )

        output = self.CustomLayerNorm1(att + data, training = is_training)
        
        ffn = self.FeedForward(output, training=is_training)
        ffn = self.CustomDropout(ffn, training = is_training)

        encoder_layer = self.CustomLayerNorm2(ffn + output, training=is_training)

        return encoder_layer

class CustomEncoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        *,
        number_layers, 
        embedded_dim,
        number_heads, 
        fc_dims,
        row, col,
        drop_rate=0.1,
        layernorm_eps=1e-6
    ):
        super().__init__()

        self.embedded_dim = embedded_dim
        self.number_layers = number_layers

        self.CustomEmbeddingLayer = tf.keras.layers.Dense(embedded_dim, activation='relu')

        self.CustomPositonalEncode = pos_encode_2D(row, col, embedded_dim)

        self.CustomEncoderLayer = [CustomEncoderBlock(
                                embedded_dim=embedded_dim,
                                number_heads = number_heads,
                                drop_rate=drop_rate,
                                layernorm_eps=layernorm_eps,
                                fc_dims=fc_dims
                                ) for _ in range(number_layers)]
        
        self.CustomDropout = tf.keras.layers.Dropout(drop_rate)
    def call(self, data, is_training, padding_mask=None):
        sequence_len = tf.shape(data)[1]

        data = self.CustomEmbeddingLayer(data)
        data += self.CustomPositonalEncode[:, :sequence_len, :]

        data = self.CustomDropout(data, training=is_training)

        for i in range(self.number_layers):
            data = self.CustomEncoderLayer[i](data, is_training, padding_mask)
        
        return data

class CustomDecoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                *,
                embedded_dim,
                number_heads,
                fc_dims,
                drop_rate=0.1,
                layernorm_eps=1e-6
                ):
        super().__init__()
        self.MaskedMHA = tf.keras.layers.MultiHeadAttention(
            num_heads=number_heads,
            key_dim=embedded_dim,
            dropout=drop_rate
        )

        self.CrossMHA = tf.keras.layers.MultiHeadAttention(
            num_heads=number_heads,
            key_dim=embedded_dim,
            dropout=drop_rate
        )

        self.FeedForward = point_wise_ffn(embedded_dim=embedded_dim, fc_dims=fc_dims)

        self.CustomLayerNorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.CustomLayerNorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.CustomLayerNorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.CustomDropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, data, output_encoder, is_training, look_ahead_mask=None, padding_mask=None):
        masked_att, mask_att_weights = self.MaskedMHA(
            query=data,
            value=data,
            key=data,
            attention_mask=look_ahead_mask,
            return_attention_scores=True,
            training=is_training
        )
        output = self.CustomLayerNorm1(masked_att + data)

        cross_att, cross_att_weights = self.CrossMHA(
            query=output, 
            value=output_encoder,
            key=output_encoder,
            attention_mask=padding_mask,
            return_attention_scores=True,
            training=is_training
        )

        second_output = self.CustomLayerNorm2(cross_att + output)

        ffn = self.FeedForward(second_output)
        ffn = self.CustomDropout(ffn, training=is_training)

        final_output = self.CustomLayerNorm3(ffn + second_output)

        return final_output, mask_att_weights, cross_att_weights

class CustomDecoder(tf.keras.layers.Layer):
    def __init__(self,
            *,
            number_layers,
            embedded_dim,
            number_heads,
            fc_dims,
            vocab_length,
            drop_rate=0.1,
            layernorm_eps=1e-6
            ):
        super(CustomDecoder, self).__init__()

        self.embedded_dim = embedded_dim
        self.number_layers = number_layers

        self.CustomEmbeddingLayer = tf.keras.layers.Embedding(
            vocab_length,
            embedded_dim,
            mask_zero=True
        )

        self.CustomPositionalEncode = pos_encode_1D(max_length, embedded_dim)

        self.CustomDecoderLayers = [
            CustomDecoderBlock(
                embedded_dim=embedded_dim,
                number_heads=number_heads,
                fc_dims=fc_dims,
                drop_rate=drop_rate,
                layernorm_eps=layernorm_eps
            ) for _ in range(number_layers)
        ]

        self.CustomDropout = tf.keras.layers.Dropout(drop_rate)
    
    def call(self, data, encode_output, is_training, look_ahead_mask, padding_mask):
        sequence_len = tf.shape(data)[1]
        att_weights = {}

        data = self.CustomEmbeddingLayer(data)
        data *= tf.math.sqrt(tf.cast(self.embedded_dim, tf.float32))
        data += self.CustomPositionalEncode[:, :sequence_len, :]
        
        data = self.CustomDropout(data, training=is_training)

        for i in range(self.number_layers):
            data, block1, block2 = self.CustomDecoderLayers[i](data, encode_output, is_training, look_ahead_mask, padding_mask)

            att_weights[f'decoder_layers{i+1}_first_block'] = block1
            att_weights[f'decoder_layers{i+1}_second_block'] = block2

        return data, att_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.cast(tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2), tf.float32)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



class CustomTransformer(tf.keras.Model):
    def __init__(self,
                *,
                number_layers,
                embedded_dim,
                number_heads,
                fc_dims,
                row, col,
                vocab_length,
                drop_rate=0.1,
                layernorm_eps=1e-6
            ):
        super().__init__()

        self.encoder = CustomEncoder(
            number_layers=number_layers,
            embedded_dim=embedded_dim,
            number_heads=number_heads,
            fc_dims=fc_dims,
            row=row,
            col=col,
            drop_rate=drop_rate,
            layernorm_eps=layernorm_eps
        )

        self.decoder = CustomDecoder(
            number_layers=number_layers,
            embedded_dim=embedded_dim,
            number_heads=number_heads,
            fc_dims=fc_dims,
            vocab_length=vocab_length,
            drop_rate=drop_rate,
            layernorm_eps=layernorm_eps
        )
        
        self.output_layer = tf.keras.layers.Dense(vocab_length)

        learning_rate = CustomSchedule(embedded_dim)
        self.optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def call(self, input, output, is_training, encode_padding_mask, look_ahead_mask, decode_padding_mask):
        encode_output = self.encoder(input, is_training, encode_padding_mask)

        decode_output, att_weights = self.decoder(output, encode_output, is_training, look_ahead_mask, decode_padding_mask)

        final_output = self.output_layer(decode_output)

        return final_output, att_weights
    @tf.function
    def train_step(self, input, output):
        target_input = output[:, :-1]
        true_value = output[:, 1:]

        look_ahead_mask = make_look_ahead_mask(tf.shape(target_input)[1])
        decode_padding_mask = None
        
        with tf.GradientTape() as tape:
            predictions, _ = self.call(input, target_input, True, None, look_ahead_mask, decode_padding_mask)
            loss = loss_function(true_value, predictions)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
