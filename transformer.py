import tensorflow as tf 
from typing import List
from layer import FeedForwardNetwork, ResidualNormalizationWrapper, LayerNormalization
import copy
import math
import six
import json
import re
##
#from embedding import TokenEmbedding, AddPositionalEncoding, get_shape_list
##
import embedding
import attention
import numpy as np
#print("version is:", tf.__version__)

PAD_ID = 0

class TConfig(object):
    def __init__(
            self,
            vocab_size: int,
            hopping_num: int = 4,
            head_num: int = 4,
            hidden_dim: int = 512,
            dropout_rate: float = 0.02,
            max_length: int = 200,
            initializer_range: float=0.02) -> None:

        self.vocab_size = vocab_size
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        config = TConfig(vocab_size = None)
        for (key,value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Transformer(tf.keras.models.Model):
    def __init__(self,
                 config,
                 input_ids,
                 is_training,
                 input_mask=None,
    ):

        super().__init__()
        config = copy.deepcopy(config)
        hid_dim = config.hidden_dim
        input_shape = embedding.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype = tf.int32)

        if not is_training:
           config.dropout_rate = 0.0

        enc_attention_mask = create_enc_attention_mask(input_ids, input_mask)
        encoder_s= Encoder(
                input_ids = input_ids,
                vocab_size = config.vocab_size,
                hopping_num = config.hopping_num,
                head_num = config.head_num,
                hidden_dim = config.hidden_dim,
                dropout_rate = config.dropout_rate,
                max_length = config.max_length,
                initializer_range=config.initializer_range,
                self_attention_mask = enc_attention_mask,
                is_training = is_training,
                )
        self.encoder = encoder_s.out_attention()
        self.sequence_output = self.encoder[-1]
        with tf.variable_scope("pooler"):
            first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :],axis=1)
            self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    hid_dim,
                    activation = tf.tanh,
                    kernel_initializer= creat_initializer(config.initializer_range)
            )

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

def create_enc_attention_mask(encoder_input, to_mask):
    from_shape = embedding.get_shape_list(encoder_input, expected_rank = [2,3])
    batch_size = from_shape[0] 
    length = from_shape[1]
    to_shape = embedding.get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]
    to_mask = tf.cast(
    tf.reshape(to_mask, [batch_size,1,to_seq_length]),tf.float32)
    broadcast_ones = tf.ones( shape=[batch_size, length,1], dtype=tf.float32)
    mask = broadcast_ones * to_mask
    return mask


def creat_initializer(initializer_range =0.02):
    t = tf.truncated_normal_initializer(stddev=initializer_range,seed=1)
    return t 

def layer_norm(input_tensor, name= None):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor,begin_norm_axis=-1, begin_params_axis=-1,scope=name)

def gelu(input_tensor):
    cdf = 0.5 * (1.0+tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf

def reshape_to_matrix(input_tensor):
    ndims = input_tensor.shape.ndims
    if ndims <2:
      raise ValueError("Input tensor must have at least rank 2. Shape = %s" % (input_tensor.shape))
   
    if ndims ==2:
      return input_tensor
    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor,[-1,width])
    return output_tensor

def reshape_from_matrix(output_tensor,orig_shape_list):
    if len(orig_shape_list)==2:
      return outpu_tensor
    output_shape = embedding.get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor,orig_dims + [width])

def droopout(input_tensor,dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output

class Encoder(tf.keras.models.Model):

    def __init__(
            self,
            input_ids,
            vocab_size:int,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            max_length: int,
            initializer_range:float,    
            #
            is_training,
            self_attention_mask,
            #
    ):

        super().__init__()
        self.attention_mk = self_attention_mask
        self.hopping_num = hopping_num
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.training = is_training
        self.initializer_range = initializer_range
         
        te = embedding.TokenEmbedding(input_ids, vocab_size, hidden_dim)
        self.token_embedding = te.token_embedding

        #self.lookup_table = te.lookup_tabel
        #(self.token_embedding, self.lookup_table) = embedding.TokenEmbedding(input_ids, vocab_size, hidden_dim)
        ##
        
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def out_attention(self):
   
        embedded_input = self.token_embedding
        embedded_input = embedding.AddPositionalEncoding(embedded_input, self.dropout_rate).positional

        a = embedding.get_shape_list(embedded_input,expected_rank=3)
        prev_output = self.input_dropout_layer(embedded_input,training=self.training)

        all_layer_outputs = []
        for _ in range(self.hopping_num):
            with tf.variable_scope("layer_%d" % _):
                layer_input = prev_output
                with tf.variable_scope("attention"):
                    attention_heads = []

                    attention_H =attention.MultiheadAttention(
                                      layer_input, 
                                      layer_input,
                                      self.attention_mk,
                                      self.hidden_dim, 
                                      self.head_num, 
                                      self.dropout_rate,
                                      self.training
                                     )
                    attention_head = attention_H.attention_layer()
                    attention_heads.append(attention_head)
                    attention_output = None
                    if len(attention_heads)==1:
                       attention_output = attention_heads[0]
                    else:
                       attention_output = tf.concat(attention_heads, axis=-1)
                    with tf.variable_scope("output"):
                         attention_output = tf.layers.dense(
                                          attention_output,
                                          self.hidden_dim,
                                          kernel_initializer = creat_initializer(self.initializer_range))
                         attention_output = droopout(attention_output,self.dropout_rate)
                         attention_output = layer_norm(attention_output + layer_input)
                intermediate_size = 1024
                
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(
                        attention_output,
                        intermediate_size,
                        activation=gelu,
                        kernel_initializer = creat_initializer(self.initializer_range))

                with tf.variable_scope("output"):
                    layer_output = tf.layers.dense(
                        intermediate_output,
                        self.hidden_dim,
                        kernel_initializer = creat_initializer(self.initializer_range))
                    layer_output = droopout(layer_output,self.dropout_rate)
                    layer_output = layer_norm(layer_output + attention_output)
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)

        final_outputs = []
        for layer_output in all_layer_outputs:
           final_output = reshape_from_matrix(layer_output,a)
           final_outputs.append(final_output)
        return final_outputs
