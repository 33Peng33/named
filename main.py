import tensorflow as tf
import numpy as np
import os, argparse, time
#from models import Model
import models1
import sys
sys.path.insert(0,'/Tagging/ops')
import transformer 
from utils import str2bool, get_logger, get_entity
from vocabss import read_data, read_dictionary, tag2label

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='Selfatteinion-crf for nlp')

parser.add_argument('-data_dir', type=str, default='data_path', help='train data source')
parser.add_argument('-test_dir', type=str, default='data_path', help='test data source')
parser.add_argument('-dev_dir', type=str, default='data_path', help='dev data source')
parser.add_argument('-output_dir', type=str, default='data_path', help='output data source')
parser.add_argument('-batch_size',type=int, default=64,help='#sample of each minibatch')
parser.add_argument('-max_seq_len',type=int, default=120, help='max input sequence length after WordPiece')
parser.add_argument('-hidden_dim',type=int, default=512,help='#dim of hidden state')
parser.add_argument('-optimizer',type=str, default='Adam',help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('-epoch',type=int, default=15)
parser.add_argument('-num_train_epochs',type=float,default=10,help='Total number of training epochs to perform')
parser.add_argument('-warmup_proportion',type=float,default=0.1)
parser.add_argument('-lr',type=float, default=1e-5,help='The initial learning rate for Adam')
parser.add_argument('-dropout_rate',type=float, default=0.02, help='Dropout rate')
parser.add_argument('-clip', type=float, default=0.5,help='Gradient clip')
parser.add_argument('-mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('-demo_model', type=str, default='1521112368',help='model for test and demo')
parser.add_argument('-config_file', type=str,default='config_path',help='json data')
parser.add_argument('-init_checkpoint',type=str,default='data_path',help='check point')
parser.add_argument('-do_train', action='store_false',default=True)
parser.add_argument('-do_eval', action='store_false', default=True)
parser.add_argument('-do_predict', action='store_false', default=True)
parser.add_argument('-save_checkpoints_steps', type=int, default=500,help='save_checkpoints_steps')
parser.add_argument('-save_summary_steps',type=int,default=500,help='save_summary__steps')


args= parser.parse_args()


word2id = read_dictionary(os.path.join('.', args.data_dir, 'word2id.pkl'))

#tconfig = transformer.TConfig.from_json_file(os.path.join(args.config_file,'config.json')) 

tconfig = transformer.TConfig.from_json_file(args.config_file)

if args.mode != 'demo':
    train_path = os.path.join('.', args.data_dir, 'train.txt')
    test_path = os.path.join('.', args.test_dir, 'test.txt')
    dev_path = os.path.join('.', args.dev_dir,'dev.txt')
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    dev_data = read_data(dev_path)
    test_size = len(test_data)

paths = {}

timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.data_dir+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)

print("data loaded.")

if args.mode == 'train':
    
    models1.train(args, train=train_data, test=test_data, dev=dev_data, vocab=word2id,tag2label=tag2label,tconfig=tconfig)

elif args.mode == 'test':
    ckpt_file = tf.train.latest_chekpoint(model_path)
    paths['model_path'] = ckpt_file
    mode = Model(args, tag2label, word2id, paths, tconfig=tconfig)
    model.build_graph()
    model.test(test_data)

elif args.mode == 'demo':
    ckpt_file=tf.train.latest_checkpoint(model_path)
    paths['model_path'] = ckpt_file
    model = Model(args, tag2label, word2id, paths, tconfig=tconfig)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=tconfig) as sess:
        print("=========demo=========")
        saver.restore(sess, ckpt_file)
        while(1):
            print("input sentece")
            demo_sent=input()
            if demo_sent=='' or demo_sent.isspace():
                print("bye")
                break
            else:
                demo_sent =list(demo_sent.strip())
                demo_data = [(demo_sent,['O']*len(demo_sent))]
                print(demo_data)
                
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
