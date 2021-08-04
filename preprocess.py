import os,sys
sys.path.insert(0,'/aspect_extract2/ops/')
from vocab import vocab_build
import codecs

data = "/aspect_extract2/dataset/train.txt"
save = "/aspect_extract2/dataset/word2id.pkl"

vocab_build(save, data,0)
