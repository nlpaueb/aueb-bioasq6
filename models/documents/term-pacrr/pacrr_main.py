import os
import sys
import json
import uuid
import copy
import keras
import shutil
import pickle
import gensim
import random
import argparse
import datetime
import subprocess

import numpy as np
import tensorflow as tf

from os import listdir
from tqdm import tqdm
from random import shuffle
from os.path import isfile, join
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors

from pacrr_model import PACRR
from utils.pacrr_utils import pacrr_train, pacrr_predict, map_term2ind, produce_pos_neg_pairs, produce_reranking_inputs, shuffle_train_pairs
from utils.bioasq_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('-config', dest='config_file')
parser.add_argument('-params', dest='params_file')
parser.add_argument('-log', dest='log_name', default='run')
args = parser.parse_args()

config_file = args.config_file
params_file = args.params_file

with open(config_file, 'r') as f:
	config = json.load(f)  
with open(params_file, 'r') as f:
	model_params = json.load(f)  

topk = 100

data_directory = '../../../bioasq6_data'

bm25_data_path_train = os.path.join(data_directory, 'pacrr_split', 'bioasq6_bm25_top{0}.pacrr_train.pkl'.format(topk, topk))
docset_path_train = os.path.join(data_directory, 'pacrr_split', 'bioasq6_bm25_docset_top{0}.pacrr_train.pkl'.format(topk, topk))

bm25_data_path_dev = os.path.join(data_directory, 'pacrr_split', 'bioasq6_bm25_top{0}.pacrr_dev.pkl'.format(topk, topk))
docset_path_dev = os.path.join(data_directory, 'pacrr_split', 'bioasq6_bm25_docset_top{0}.pacrr_dev.pkl'.format(topk, topk))

bm25_data_path_test = {}
docset_path_test = {}
qrels_test = {}

# Test batches
for i in range(1, 6):
	bm25_data_path_test[i] = os.path.join(data_directory, 'test_batch_{0}'.format(i), 'bioasq6_bm25_top{0}.test.pkl'.format(topk))
	docset_path_test[i] = os.path.join(data_directory, 'test_batch_{0}'.format(i), 'bioasq6_bm25_docset_top{0}.test.pkl'.format(topk))
	qrels_test[i] = os.path.join(data_directory, 'test_batch_{0}'.format(i), 'BioASQ-task6bPhaseB-testset{0}'.format(i))

w2v_path = config['WORD_EMBEDDINGS_FILE']
idf_path = config['IDF_FILE']


with open(bm25_data_path_train, 'rb') as f:
	data_train = pickle.load(f)

with open(docset_path_train, 'rb') as f:
	docset_train = pickle.load(f)

with open(bm25_data_path_dev, 'rb') as f:
	data_dev = pickle.load(f)

with open(docset_path_dev, 'rb') as f:
	docset_dev = pickle.load(f)

with open(idf_path, 'rb') as f:
	idf = pickle.load(f)

print('All data loaded. Pairs generation started..')

# map each term to an id
term2ind = map_term2ind(w2v_path)

# Produce Pos/Neg pairs for the training subset of queries.
print('Producing Pos-Neg pairs for training data..')
train_pairs = produce_pos_neg_pairs(data_train, docset_train, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq, 2015)

# Produce Pos/Neg pairs for the development subset of queries.
print('Producing Pos-Neg pairs for dev data..')
dev_pairs = produce_pos_neg_pairs(data_dev, docset_dev, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq, 2016)

# Produce reranking inputs for the development subset of queries.
print('Producing reranking data for dev..')
dev_reranking_data = produce_reranking_inputs(data_dev, docset_dev, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq, 2016)

#Random shuffle training pairs
train_pairs = shuffle_train_pairs(train_pairs)

retr_dir = os.path.join('logs', args.log_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print(retr_dir)
os.makedirs(os.path.join(os.getcwd(), retr_dir))

json_model_params = copy.deepcopy(model_params)
json_model_params['embed'] = []
with open(retr_dir+ '/{0}'.format(params_file.split('/')[-1]), 'w') as f:
	json.dump(json_model_params, f, indent=4)

metrics = ['map', 'gmap', 'f1']

res = pacrr_train(train_pairs, dev_pairs, dev_reranking_data, term2ind, config, model_params, metrics, retr_dir, doc_id_prefix='http://www.ncbi.nlm.nih.gov/pubmed/', keep_topk=10)

print('Training finished.')

for i in range(1, 6):
	print('\n===========================')
	print('Evaluating on test batch {0}'.format(i))
	print('===========================')

	with open(bm25_data_path_test[i], 'rb') as f:
		data_test = pickle.load(f)
	with open(docset_path_test[i], 'rb') as f:
		docset_test = pickle.load(f)

	# Produce reranking inputs for the test subset of queries.
	print('Producing reranking data for test..')
	test_reranking_data = produce_reranking_inputs(data_test, docset_test, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq)
	pacrr_predict(*res, test_reranking_data, term2ind, qrels_test[i], model_params, metrics, retr_dir, doc_id_prefix='http://www.ncbi.nlm.nih.gov/pubmed/', keep_topk=10, batch_num=i)
