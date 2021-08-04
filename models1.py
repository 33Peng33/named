import os, sys
import tensorflow as tf
import codecs,pickle,collections
import optimization
import early_stopping
from set_report import set_logger,get_assignment_map_from_checkpoint
from self_crf import SELF_CRF
from tensorflow.contrib.layers.python.layers import initializers
sys.path.insert(0,'/Tagging/ops/')
import vocabss

logger = set_logger('Training')

def Model(args,tag2label,num_train_steps,num_warmup_steps,tconfig):
    batch_size = args.batch_size
    lr = args.lr
    num_tags = len(tag2label)
    tconfig = tconfig

    def model_fn(features,labels, mode, params):
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info(" name =%s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        label_ids = features["label_ids"]
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        total_loss, logits, trans,pred_ids = create_model(tconfig = tconfig, is_training=is_training, input_ids=input_ids, input_mask=input_mask, labels=label_ids, num_tags = num_tags)
        tvars = tf.trainable_variables()

        """     
        if init_checkpoint :
           (assignment_map, initialized_variable_names) = \
                       get_assignment_map_from_checkpoint(tvars,init_checkpoint)
           tf.train.init_from_checkpoint(init_checkpoint,assignment_map)
        """
        output_spec = None
    

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                       total_loss, lr, num_train_steps, num_warmup_steps, False)
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                        hook_dict, every_n_iter = args.save_summary_steps)
            output_spec = tf.estimator.EstimatorSpec(
                     mode = mode,
                     loss = total_loss,
                     train_op = train_op,
                     training_hooks = [logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_ids, pred_ids):
                return {
                   "eval_loss": tf.metrics.mean_squared_error(labels=label_ids,predictions = pred_ids),
                  }
            eval_metrics = metric_fn(label_ids,pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                       mode = mode,
                       loss = total_loss, 
                       eval_metric_ops = eval_metrics
                      )
        else:
              output_spec = tf.estimator.EstimatorSpec(
                        mode = mode,
                        predictions = pred_ids
                       )
        return output_spec
    return model_fn


def train(args,train,test,dev,vocab,tag2label,tconfig):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(
        log_device_placement = False,
        inter_op_parallelism_threads =0,
        intra_op_parallelism_threads =0,
        allow_soft_placement=True,
        gpu_options = gpu_options)
    
#    session_config.gpu_options.per_process_gpu_memory_fraction=0.4

    run_config = tf.estimator.RunConfig(
        model_dir = args.output_dir,
        save_summary_steps = 500,
        save_checkpoints_steps=500,
        session_config=session_config
       )

    num_train_steps = None
    num_warmup_steps = None


    if args.do_train and args.do_eval:
        num_train_steps = int(len(train)*1.0 / args.batch_size * args.epoch)
    if num_train_steps <1:
        raise AttributeError ('training data is so small')
    num_warmup_steps = int(num_train_steps + args.warmup_proportion)

    logger.info("***Running training ***")
    logger.info("  Num examples = %d", len(train))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    model_fn = Model(args=args,tag2label=tag2label,num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps,tconfig=tconfig)
    params = {'batch_size':args.batch_size}
    
    estimator = tf.estimator.Estimator(
            model_fn,
            params=params,
            config=run_config)

    if args.do_train and args.do_eval:
        train_file = os.path.join(args.output_dir,"train.tf_record")
        if not os.path.exists(train_file):
               filed_to_features(train, args.batch_size, args.max_seq_len, vocab, tag2label,train_file)
        train_input_fn = file_based_builder(
              input_file = train_file,
              seq_length= args.max_seq_len,
              is_training = True,
              drop_remainder=True) 
        eval_file = os.path.join(args.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
               filed_to_features(test,args.batch_size,args.max_seq_len,vocab,tag2label, eval_file)
        
        eval_input_fn = file_based_builder(
             input_file = eval_file,
             seq_length = args.max_seq_len,
             is_training=False,
             drop_remainder=False)

        early_stopping_hook = early_stopping.stop_if_no_decrease_hook(
             estimator=estimator,
             metric_name='loss',
             max_steps_without_decrease=num_train_steps,
             eval_dir=None,
             min_steps=0,
             run_every_secs=None,
             run_every_steps=args.save_checkpoints_steps)
 
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
    
    if args.do_predict:
        dev_file = os.path.join(args.output_dir,"dev.tf_record")
        filed_to_features(dev,args.batch_size,args.max_seq_len,vocab,tag2label,dev_file)
        dev_input_fn = file_based_builder(
            input_file = dev_file,
            seq_length=args.max_seq_len,
            is_training = False,
            drop_remainder = False)

        output_predict_file = os.path.join(args.output_dir,"label_test.txt")
        result = estimator.predict(input_fn = dev_input_fn)
         
        id2label ={value:key for key,value in tag2label.items()}

        def result_to_pair(writer):
            for predict_line, prediction in zip(dev,result):
                idx=0
                line = ''
                line_token=predict_line[0].split(' ')
                label_token=predict_line[1].split(' ')
                len_seq = len(label_token)
                for id_ in prediction:
                    if idx >= len_seq: 
                        break
                    if id_ == 0:
                        continue
                    curr_labels = id2label[id_]
                    try:
                        line += line_token[idx] + ' ' + label_token[idx] + ' ' +curr_labels + '\n'
                    except Exception as e:
                        logger.info(e)
                        logger.info(predict_line[0])
                        logger.info(predict_line[1])
                        line = ''
                        break
                    idx += 1
                writer.write(line + '\n')
        with open(output_predict_file,'w') as writer:
            result_to_pair(writer)
        import colleval
        eval_result = colleval.return_report(output_predict_file)
        print(''.join(eval_result))
        with open(os.path.join(args.output_dir,'predict_score.txt'),'a',encoding='utf-8') as fd:
            fd.write(''.join(eval_result))
                    
def conver_single_example(example, batch_size, max_sequences,vocab, tag2label):
 
    id_data = vocabss.get_id_data(example, vocab, tag2label)
    for seqs, label in id_data:
        
        input_mask = [1]*len(seqs)
        while len(seqs)<max_sequences:
              seqs.append(0)
              label.append(0)
              input_mask.append(0)

    assert len(seqs) == max_sequences
    assert len(label) == max_sequences
    assert len(input_mask) == max_sequences

    feature = InputFeatures(
             input_ids = seqs,
             input_mask = input_mask,
             label_ids = label,
                   )
    return feature 

def filed_to_features(examples,batch_size,max_sequences,vocab,tag2label,output_file):
    """
    write to TF_Record
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    
    for example in examples:
        feature = conver_single_example(example,batch_size,max_sequences,vocab,tag2label)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features= tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length],tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length],tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length],tf.int64),
             }

    def _decode_record(record,name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype==tf.int64:
               t = tf.to_int32(t)
               example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
           d = d.repeat()
           d = d.shuffle(buffer_size = 300)

        a=tf.data.experimental.map_and_batch(lambda record:_decode_record(record, name_to_features),
                    batch_size = batch_size,
                    num_parallel_calls=3,
                    drop_remainder = drop_remainder)
        d= d.apply(a)
        d = d.prefetch(buffer_size=4)
        return d
    return input_fn

def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path)):
        logger.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
               continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last

def adam_filter(model_path):
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return 
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path,last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess,os.path.join(model_path,'model.ckpt'))

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

class InputFeatures(object):
      def __init__(self,input_ids,input_mask,label_ids,):
          self.input_ids = input_ids
          self.input_mask = input_mask
          self.label_ids = label_ids
