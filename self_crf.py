import tensorflow as tf
from tensorflow.contrib import crf

class SELF_CRF(object):
    def __init__(self, embedded_chars,dropout_rate,initializers,
                 num_tags, seq_length, labels, lengths, is_training):
        
        self.dropout_rate=dropout_rate
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.labels = labels
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training
        self.num_tags = num_tags
        self.lengths = lengths

    def add_crf_layer(self):
        
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        logits= self.project_crf_layer(self.embedded_chars)
        loss, trans = self.crf_layer(logits)
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans,sequence_length=self.lengths)
        return (loss, logits, trans, pred_ids)

    def project_crf_layer(self, embedding_chars, name=None):

        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_tags], dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b",shape=[self.num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())
                
                output = tf.reshape(self.embedded_chars, shape=[-1, self.embedding_dims])
                pred = tf.tanh(tf.nn.xw_plus_b(output,W,b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_tags])

    def crf_layer(self, logits):

        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable("transitions",
                    shape=[self.num_tags, self.num_tags],
                    initializer= self.initializers.xavier_initializer())
            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                        inputs=logits,
                        tag_indices=self.labels,
                        transition_params=trans,
                        sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans
