import dynet as dy
import pickle
import heapq
import utils
import numpy
import random
import model
import sys

print('Loading Data')
dataloc = '../../../bioasq6_data/'
with open(dataloc + 'test_batch_' + sys.argv[2] +
          '/bioasq6_bm25_top100.test.pkl', 'rb') as f:
  data = pickle.load(f)
with open(dataloc + 'test_batch_' + sys.argv[2] +
          '/bioasq6_bm25_docset_top100.test.pkl', 'rb') as f:
  docs = pickle.load(f)

words = {}
utils.GetWords(data, docs, words)

model = model.Model(dataloc, words)
model.Load(sys.argv[1])

print('Making preds')
json_preds = {}
json_preds['questions'] = []
num_docs = 0
for i in range(len(data['queries'])):
  num_docs += 1
  dy.renew_cg()

  qtext = data['queries'][i]['query_text']
  qwds, qvecs, qconv = model.MakeInputs(qtext)

  rel_scores = {}
  for j in range(len(data['queries'][i]['retrieved_documents'])):
    doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
    dtext = (docs[doc_id]['title'] + ' <title> ' +
             docs[doc_id]['abstractText'])
    dwds, dvecs, dconv = model.MakeInputs(dtext)
    bm25 = data['queries'][i]['retrieved_documents'][j]['norm_bm25_score']
    efeats = model.GetExtraFeatures(qtext, dtext, bm25)
    efeats_vec = dy.inputVector(efeats)
    score = model.GetQDScore(qwds, qconv, dwds, dconv, efeats_vec)
    rel_scores[j] = score.value()

  top = heapq.nlargest(10, rel_scores, key=rel_scores.get)
  utils.JsonPredsAppend(json_preds, data, i, top)
  dy.renew_cg()

utils.DumpJson(json_preds, 'abel_test_preds_batch' + sys.argv[2] + '.json')
print('Done')
