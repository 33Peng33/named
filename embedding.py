import tensorflow as tf
import math
import six
import numpy as np


PAD_ID = 0


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_ids, vocab_size, embedding_dim, dtype=tf.float32):
        super().__init__()
        self.input_ids = input_ids
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype_ = dtype

        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(self.input_ids, axis=[-1])

        self.lookup_tabel = tf.get_variable(
                name='token_embedding',
                shape=[self.vocab_size,self.embedding_dim],
                dtype=self.dtype_,
                initializer=tf.random_normal_initializer(0., self.embedding_dim ** -0.5)
         )


        output = tf.nn.embedding_lookup(self.lookup_tabel, self.input_ids)

        self.token_embedding = output * self.embedding_dim **0.5
	

class AddPositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, input_tensor, dropout_rate=0.1):
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        max_length = input_shape[1]
        depth = input_shape[2]
        fl_type = input_tensor.dtype

        depth_counter = tf.range(depth) // 2*2
        depth_matrix = tf.tile(tf.expand_dims(depth_counter, 0), [max_length,1])
        depth_matrix = tf.pow(10000.0, tf.cast(depth_matrix / depth, fl_type))

        phase = tf.cast(tf.range(depth) % 2, fl_type)* math.pi /2
        phase_matrix = tf.tile(tf.expand_dims(phase,0), [max_length,1])

        pos_counter = tf.range(max_length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1),[1,depth]), fl_type)

        positional_encoding = tf.sin(pos_matrix / depth_matrix + phase_matrix)
        positional_encoding = tf.tile(tf.expand_dims(positional_encoding, 0), [batch_size,1,1])
        self.positional = input_tensor + positional_encoding

def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor,expected_rank,name)

    shape = tensor.shape.as_list()
    non_static_indexes=[]

    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank]=True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True
    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
                "For the tensor `%s` in scope `%s`, the actual rank "
                "`%d` (shape = %s) is not equal to the expected rank `%s`" %
                (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
