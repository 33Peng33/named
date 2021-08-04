import sys, os, pickle
import numpy as np
import gensim
from gensim import corpora
import codecs

"""
tag2label = {"O": 0,
             "B-NP":1, "I-NP":2,
             "B-VP":3, "I-VP":4,
             "B-PP":5, "I-PP":6,
             "B-SBAR":7, "I-SBAR":8,
             "B-ADVP":9, "I-ADVP":10,
             "B-ADJP":11, "I-ADJP":12,
             "B-CONJP":13, "I-CONJP":14,
             "B-INTJ":15, "I-INTJ":16,
             "B-LST":17, "I-LST":18,
             "B-PRT":19, "I-PRT":20
             }
"""
tag2label = {"O":0, "B-AP":1, "I-AP":2}

def read_data(data_path):
    data = []
    with open(data_path,'r',encoding='utf-8') as fr:
        lines= fr.readlines()
    for line in lines:
        [char,label] = line.strip('\n').split(' ||| ')
        data.append([char,label])
#    print(data)
    return data


def read1_data(data_path):
    data = []
    with codecs.open(data_path,'r',encoding='utf-8') as fr:
        lines = []
        sent_, tag_ = [], []
        for line in fr:
            if len(line.strip().split(' ')) ==2:
                [char,label] = line.strip().split(' ')
                sent_.append(char)
                tag_.append(label)
                #print(char)
      #        print(label)
            else:
                if len(line.strip())==0:
                    data.append([' '.join(sent_),' '.join(tag_)])
                    sent_,tag_=[], []

    #print(data)
    return data 
         
def vocab_build(vocab_path, data_path):
    data = read_data(data_path)
    word2id = {}
    for num,sent_ in enumerate(data):
        tokens = [[token for token in sentence.split()] for sentence in sent_]
        if num == 0:
            gensim_dictionary = corpora.Dictionary(tokens)
        else:
            gensim_dictionary.add_documents(tokens)
    word2id = gensim_dictionary.token2id
    with open(vocab_path,'wb') as fw:
        pickle.dump(word2id,fw,0)
    fw.close()

def vocab_upgrade(word,word2id):
    id_s=max(word2id.values())
    id_s += 1
    dic = {word:id_s}
    word2id.update(dic)
    return word2id
   
def sentence2id(sent, word2id):

    sentence_id = []
    for word in sent:
        if word not in word2id:
           vocab_upgrade(word,word2id)
        sentence_id.append(word2id[word])
    return sentence_id

def read_dictionary(data_path):
    data_path = os.path.join(data_path)
    with open(data_path, 'rb') as fr:
        word2id = pickle.load(fr)
    fr.close()
    return word2id

def pad_sequences(sequences, pad=0):
    max_len = max(map(lambda x : len(x),sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad] * max(max_len -len(seq),0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def get_token(data, batch_size, vocab,tag2label):
    assert len(data)==2
    for sequences in data:
        token = data[0]
        label = data[-1]

def get_id_data(data,vocab,tag2label):
    seqs,labels=[],[]
    sent_ = data[0].split(' ')
    tag_ = data[-1].split(' ')
    sent_ = sentence2id(sent_,vocab)
    label_ = [tag2label[tag] for tag in tag_]
    seqs.append(sent_)
    labels.append(label_)
    
    id_data= zip(seqs,labels)
    return id_data

def batch_yield(data, batch_size, vocab,tag2label):
    seqs, labels = [], []
    for (sent_,tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)
    
    batch_m = (seqs,labels)
 #   print("129 vocab.py"+"\n") 
#    print(seqs)
    if len(seqs) !=0:
        yield batch_m

