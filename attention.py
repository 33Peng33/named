import tensorflow as tf
import embedding
import math

class MultiheadAttention(tf.keras.models.Model):
    def __init__(self,
                 query,
                 memory,
                 attention_mask, 
                 hidden_dim: int, 
                 head_num: int,
                 dropout_rate: float,
                 training: bool):

        super().__init__()
        self.hidden_dim= hidden_dim
        self.head_num = head_num
        self.dropout_rate= dropout_rate
        
        self.query = query
        self.memory = memory
        self.attention_mask = attention_mask
        self.training = training


    def attention_layer(self):
        
        in_put = self.query
        memory = self.memory
        attention_mask = self.attention_mask
        training = self.training

        self.q_dense_layer= tf.layers.dense(in_put,self.hidden_dim, activation=None, name='q_dense_layer',kernel_initializer=self.create_initializer(0.02))
        self.k_dense_layer= tf.layers.dense(memory,self.hidden_dim, activation=None, name='k_dense_layer',kernel_initializer=self.create_initializer(0.02))
        self.v_dense_layer= tf.layers.dense(memory,self.hidden_dim, activation=None, name='v_dense_layer',kernel_initializer=self.create_initializer(0.02))
        q= self.q_dense_layer
        k= self.k_dense_layer
        v= self.v_dense_layer

        q= self._split_head(q)
        k= self._split_head(k)
        v= self._split_head(v)

        depth = self.hidden_dim
      #  q *= depth ** -0.5


        logit= tf.matmul(q,k,transpose_b=True)
        logit = tf.multiply(logit, 1.0 / math.sqrt(float(self.head_num)))
        
        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask,axis=[1])
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            logit += adder

        attention_weight = tf.nn.softmax(logit, name='attention_weight')
        attention_weight = self.dropout(attention_weight, self.dropout_rate)
        a= attention_weight.get_shape().as_list()
        b = v.get_shape().as_list()  
        attention_output= tf.matmul(attention_weight,v)
        attention_output= self._combine_head(attention_output)
        return attention_output


    def  _split_head(self, x: tf.Tensor) -> tf.Tensor:
        '''
        input:[batch_size,length,hidden_dim]
        output:[batch_size,head_num, length, hidden_dim/head_num]
        '''

        with tf.name_scope('split_head'):
            batch_size,length, hidden_dim = tf.unstack(tf.shape(x))
            x= tf.reshape(x, [batch_size,length,self.head_num, self.hidden_dim//self.head_num])
            return tf.transpose(x,[0,2,1,3])

    def _combine_head(self, x:tf.Tensor) -> tf.Tensor:
        '''
        input:[batch_size, head_num, length, hidden_dim/head_num]
        output:[batch_size, length, hidden_dim]
        '''

        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x,[0,2,1,3])
            context_layer = tf.reshape(x,[batch_size,length,self.hidden_dim])
          #  context_layer = tf.reshape(x,[batch_size * length, self.head_num * self.hidden_dim])
            return context_layer

    def dropout(self,input_tensor,dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        rate = 1.0 - dropout_prob
        output = tf.nn.dropout(input_tensor, rate)
        return output

    def create_initializer(self,initializer_range=0.02):
        return tf.truncated_normal_initializer(stddev=initializer_range)

def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor
