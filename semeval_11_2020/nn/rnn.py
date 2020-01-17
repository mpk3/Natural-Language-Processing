import numpy as np
import pickle

NN = '/home/mpk3/Natural_Language_Processing/semeval_11_2020/nn'
DATA = NN + '/data/clean_data.pickle'

data = np.array(pickle.load(open(DATA, 'rb')))

tok_sentences = [[feat_set[0] for feat_set in sent] for sent in data]
classes = [[feat_set[1] for feat_set in sent] for sent in data]
