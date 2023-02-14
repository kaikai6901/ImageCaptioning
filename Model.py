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

class Custom_Encoder(tf.keras.Model):
    def __init__(self,dim):
        super(Custom_Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(dim)
        
    def call(self, extracted_ft):
        extracted_ft =  self.dense(extracted_ft)
        extracted_ft =  tf.keras.activations.relu(extracted_ft, alpha=0.1)
        return extracted_ft

class Custom_Attention(tf.keras.Model):
    def __init__(self, units):
        super(Custom_Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) 
        self.W2 = tf.keras.layers.Dense(units) 
        self.V = tf.keras.layers.Dense(1) 
        self.units=units

    def call(self, features, hidden):
        new_hidden = hidden[:, tf.newaxis]
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(new_hidden))  
        att_weights = tf.keras.activations.softmax(self.V(score), axis=1) 
        context_vector = att_weights * features 
        context_vector = tf.reduce_sum(context_vector, axis=1)  
        return context_vector, att_weights
    
class Custom_Decoder(tf.keras.Model):
    def __init__(self, embedded_dim, units, vocab_size):
        super(Custom_Decoder, self).__init__()
        self.units=units
        self.attention = Custom_Attention(self.units) 
        self.embed = tf.keras.layers.Embedding(vocab_size, embedded_dim) 
        self.lstm = tf.keras.layers.LSTM(self.units,return_sequences=True,return_state=True)
        self.d1 = tf.keras.layers.Dense(self.units)
        self.d2 = tf.keras.layers.Dense(vocab_size) 
        

    def call(self,x,features, hidden):
        context_vector, attention_weights = self.attention(features, hidden) 
        embed = self.embed(x) 
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis = -1) 
        output,state, _ = self.lstm(embed)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.d2(output)
        return output, state, attention_weights
    
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
class Model():
    def __init__(self, embedded_dim, units, vocab_length, lr):
        self.encoder=Custom_Encoder(embedded_dim)
        self.decoder=Custom_Decoder(embedded_dim, units, vocab_length)
        self.word_to_index, self.index_to_word = get_text_transfrom()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr) 
        self.extraction_model = get_extraction_model()
        ## EfficientB3
        self.feature_shape = 1536
        self.attention_feature_shape = 81
    def set_checkpoint(self,checkpoint_path):
        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,
                           decoder=self.decoder,
                           optimizer = self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)
    def restore_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("restore checkpoint")

    @tf.function
    def train_proc(self, img_ts, tar):
        loss = 0
        hidden_state = self.decoder.init_state(batch_size=tar.shape[0])
        decoder_input = tf.expand_dims([self.word_to_index['start']] * tar.shape[0], 1)
        
        with tf.GradientTape() as tape:
            
            encoder_operator = self.encoder(img_ts)
            for r in range(1, tar.shape[1]) :
                predictions, hidden_state, _ = self.decoder(decoder_input, encoder_operator, hidden_state)
                loss = loss + loss_function(tar[:, r], predictions) 
                decoder_input = tf.expand_dims(tar[:, r], 1)  

        avg_loss = (loss/ int(tar.shape[1])) #avg loss per batch
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grad = tape.gradient (loss, trainable_vars) 
        self.optimizer.apply_gradients(zip(grad, trainable_vars))
        return loss, avg_loss
    @tf.function
    def test_proc(self,img_ts, tar):
        loss = 0
        hidden = self.decoder.init_state(batch_size = tar.shape[0])
        dec_input = tf.expand_dims([self.word_to_index['start']] * tar.shape[0], 1)
        with tf.GradientTape() as tape:
            encoder_op = self.encoder(img_ts)
            for r in range(1, tar.shape[1]) :
                predictions, hidden, _ = self.decoder(dec_input, encoder_op, hidden)
                loss = loss + loss_function(tar[:, r], predictions)
                dec_input = tf.expand_dims(tar[: , r], 1)
        avg_loss = (loss/ int(tar.shape[1])) #avg loss per batch
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grad = tape.gradient (loss, trainable_vars) 
        self.optimizer.apply_gradients(zip(grad, trainable_vars))                      
        return loss, avg_loss

    def calculate_loss(self,test_dataset):
        total_loss = 0
        for (batch, (img_tensor, target)) in enumerate(test_dataset) :
            batch_loss, t_loss = self.test_proc(img_tensor, target)
            total_loss = total_loss + t_loss
            avg_test_loss = total_loss
        return avg_test_loss

    def evaluate(self, image):
        attention_plot = np.zeros((max_length, self.attention_feature_shape))

        hidden = self.decoder.init_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0) 
        img_tensor_val = self.extraction_model(temp_input) 
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val) 

        dec_input = tf.expand_dims([self.word_to_index['start']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden) 
            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy() 
            if self.index_to_word[predicted_id] == 'end':
                return result, attention_plot,predictions
            elif self.index_to_word[predicted_id] != '[UNK]':
                result.append(self.index_to_word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot,predictions
        
    def generate(self,image_path):
        if "https://" in image_path:
            image_extension = image_path[-4:]
            image_path = tf.keras.utils.get_file('image' + image_extension, origin=image_path)
        result = self.evaluate(image_path)[0]
        print("Caption: " + " ".join(result))
        plt.imshow(plt.imread(image_path))
        plt.show()
