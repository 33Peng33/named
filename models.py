import os, sys
import tensorflow as tf
import numpy as np
import optimization
from self_crf import SELF_CRF
from tensorflow.contrib.layers.python.layers import initializers
sys.path.insert(0,'/Tagging/ops/')
import vocab
from utils import get_logger



class Model(object):
    def __init__(self, args, tag2label, vocab, paths, tconfig):
        super().__init__()
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.lr = args.lr
        self.dropout = args.dropout_rate
        self.clip_grad = args.clip
        self.optimizer = args.optimizer
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.tconfig = tconfig
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']

    def build_graph(self):
        self.add_placeholders()
        self.ct_model()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.input_ids = tf.placeholder(tf.int32, shape=[None,None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None,None], name='input_mask')
        self.labels = tf.placeholder(tf.int32, shape=[None,None], name='labels')
        self.sequence_lengths =tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.sequence2_lengths =tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name= 'dropout')
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def ct_model(self):
        self.total_loss, self.logtis, self.trans, self.pred_ids =create_model(
                tconfig=self.tconfig, is_training=True, input_ids=self.input_ids,
                input_mask=self.input_mask, labels=self.labels, num_tags=self.num_tags,
                dropout_rate=1.0)

      #  tvars = tf.trainable_variables()
       # self.global_step = tf.Variable(0,name="global_step",trainable=False)
        #self.train_op = optimization.create_optimizer(self.total_loss,self.lr_pl,15,5,False)
       # mode = tf.estimator.ModeKeys.TRAIN
       # hook_dict = {}
       # hook_dict['loss'] = self.total_loss
       # hook_dict['global_steps'] = tf.train.get_or_create_global_step()
       # global_step = hook_dict['global_steps']
       # logging_hook = tf.train.LoggingTensorHook(hook_dict, every_n_iter=15)
       # output_spec = tf.estimator.EstimatorSpec(mode = mode, loss=self.total_loss, train_op=self.train_op,training_hooks=[logging_hook])

    
    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer =="Adam":
                optim = tf.train.AdamOptimizer(self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.total_loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
    

    def init_op(self):
        self.init_op = tf.global_variables_initializer()
    
    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self,train,dev):
        saver = tf.train.Saver(tf.global_variables())

        session_config = tf.ConfigProto(
              log_device_placement=False,
              inter_op_parallelism_threads=0,
              intra_op_parallelism_threads=0,
              allow_soft_placement=True)

        session_config.gpu_options.per_process_gpu_memory_fraction=0.6

        with tf.Session(config=session_config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            print("models.py 106"+"\n")
            print(self.merged)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):

        num_batches = (len(train) + self.batch_size - 1) 
       
        #self.global_steps = tf.Variable(0,trainable=False)
        batches = vocab.batch_yield(train, self.batch_size, self.vocab, self.tag2label)
        for step, batch_m in enumerate(batches):
            seqs,label,mask = batch_m
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, mask, label, self.lr, self.dropout)
           # _, loss_train, summary, step_unm_ 
            _, loss_train, summaryi,step_num_ = sess.run([self.train_op, self.total_loss, self.merged,self.global_step],feed_dict=feed_dict)
            print("124 models.py")
           
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info('epoch {}, step {}, loss: {:.4}, global_step: {}'.format(epoch + 1, step + 1,loss_train, step_num))
            
            self.file_writer.add_summary(summary, step_num)
            if step + 1 == num_batches:
                saver.save(sess, self.model_path)
        
        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev,mask)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, mask, labels=None, lr=None, dropout=None):

        input_ids, seq_len_list = vocab.pad_sequences(seqs, pad=0)
        input_mask, _ = vocab.pad_sequences(mask, pad=0)
        feed_dict ={self.input_ids: input_ids,
                    self.input_mask: input_mask,
                    self.sequence_lengths: seq_len_list}
        
        if labels is not None:
            labels_, _ = vocab.pad_sequences(labels, pad=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] =lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict, seq_len_list

    def get_feed2_dict(self,seqs,labels=None,lr=None,dropout=None):

        input2_ids, seq2_len_list = vocab.pad_sequences(seqs, pad=0)
        feed2_dict ={input2_ids: input2_ids,
                     self.sequence2_lengths: seq2_len_list}
        
        if labels is not None:
            laels_2, _2 = pad_sequence(labels, pad=0)
            feed2_dict[self.labels] = labels_
        if lr is not None:
            feed2_dict[self.lr_pl] =lr
        if dropout is not None:
            feed2_dict[self.dropout_pl] = dropout
        return feed2_dict,seq2_len_list


    def dev_one_epoch(self, sess, dev,vmask):
        label_list =[]
        batchs = vocab.batch_yield(dev,self.batch_size, self.vocab, self.tag2label)
        seqss = list(batchs)[0]
        for seqs in seqss:

            label_list_ = self.predict_one_batch(sess,seqsi,mask)
            label_list.extend(label_list_)
        return label_list

    def predict_one_batch(self,sess,seqs,mask):

        feed_dict, seq_len_list = self.get_feed_dict(seqs,mask, dropout=1.0)
        pred_ids = sess.run(self.pred_ids,feed_dict=feed_dict)
        return pred_ids

    def evaluate(self, label_list, seq_len_list, data, epoch=None):

        label2tag={}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label !=0 else label

        model_predict =[]
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label_] for label_ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.jpin(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.jpin(self.result_path, 'result_metric_' + epoch_num)
        for _ in connlebal(model_predict, label_path, metric_path):
            self.logger.info(_)


def create_model(tconfig, is_training, input_ids,
                 input_mask, labels, num_tags, 
                 dropout_rate=1.0):

    import tensorflow as tf
    import transformer
    model = transformer.Transformer(
            config= tconfig, 
            input_ids=input_ids,
            input_mask = input_mask,
            is_training=is_training
                        )
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    if max_seq_length is None:
       max_seq_length = 200
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    self_crf = SELF_CRF(embedded_chars=embedding,dropout_rate=dropout_rate,
                        initializers=initializers,num_tags=num_tags,
                        seq_length=max_seq_length, labels=labels,
                        lengths=lengths, is_training=is_training)
    rst = self_crf.add_crf_layer()
    return rst

