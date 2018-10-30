from gensim.models.keyedvectors import KeyedVectors
import pickle
import numpy
import dynet as dy
import utils

class Model:
  def __init__(self, dataloc, words):
    print('Loading word vectors')
    wv = KeyedVectors.load_word2vec_format(
        dataloc + 'pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin',
        binary=True) # C binary format
    self.wv = {}
    for w in words:
      if w in wv:
        self.wv[w] = wv[w]
    wv = None

    print('Loading IDF tables')
    idf = {}
    with open(dataloc + 'IDF.pkl', 'rb') as f:
      idf = pickle.load(f)
    self.idf = {}
    for w in words:
      if w in idf:
        self.idf[w] = idf[w]
    self.max_idf = 0.0
    for w in idf:
      if idf[w] > self.max_idf:
        self.max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf %f' % self.max_idf)

    self.model = dy.ParameterCollection()
    self.trainer = dy.AdamTrainer(self.model, 0.001)

    self.conv_dim = 200
    self.W_conv = self.model.add_parameters((self.conv_dim, 3 * self.conv_dim))
    self.b_conv = self.model.add_parameters((self.conv_dim))
    self.pad = self.model.add_lookup_parameters((2, self.conv_dim))

    self.W_gate = self.model.add_parameters((1, self.conv_dim + 1))

    # MLP layer. Can optionally have multiple layers.
    self.h_size = 8
    self.in_size = self.conv_dim #+ 2
    self.W_term = self.model.add_parameters((self.h_size, self.in_size))
    self.b_term = self.model.add_parameters((self.h_size))
    self.W_term2 = self.model.add_parameters((1, self.h_size))
    self.b_term2 = self.model.add_parameters((1))

    # Final linear layer. With extra features.
    self.W_final = self.model.add_parameters((1, 5))
    self.b_final = self.model.add_parameters((1))

  def Save(self, filename):
    self.model.save(filename)

  def Load(self, filename):
    self.model.populate(filename)

  def idf_val(self, w):
    if w in self.idf:
      return self.idf[w]
    return self.max_idf

  def query_doc_overlap(self, qwords, dwords):
    # % Query words in doc.
    qwords_in_doc = 0
    idf_qwords_in_doc = 0.0
    idf_qwords = 0.0
    for qword in utils.uwords(qwords):
      idf_qwords += self.idf_val(qword)
      for dword in utils.uwords(dwords):
        if qword == dword:
          idf_qwords_in_doc += self.idf_val(qword)
          qwords_in_doc += 1
          break
    if len(qwords) <= 0:
      qwords_in_doc_val = 0.0
    else:
      qwords_in_doc_val = (float(qwords_in_doc) /
                           float(len(utils.uwords(qwords))))
    if idf_qwords <= 0.0:
      idf_qwords_in_doc_val = 0.0
    else:
      idf_qwords_in_doc_val = float(idf_qwords_in_doc) / float(idf_qwords)

    # % Query bigrams  in doc.
    qwords_bigrams_in_doc = 0
    idf_qwords_bigrams_in_doc = 0.0
    idf_bigrams = 0.0
    for qword in utils.ubigrams(qwords):
      wrds = qword.split('_')
      idf_bigrams += self.idf_val(wrds[0]) * self.idf_val(wrds[1])
      for dword in utils.ubigrams(dwords):
        if qword == dword:
          qwords_bigrams_in_doc += 1
          idf_qwords_bigrams_in_doc += (self.idf_val(wrds[0])
                                        * self.idf_val(wrds[1]))
          break
    if len(qwords) <= 0:
      qwords_bigrams_in_doc_val = 0.0
    else:
      qwords_bigrams_in_doc_val = (float(qwords_bigrams_in_doc) /
                                   float(len(utils.ubigrams(qwords))))
    if idf_bigrams <= 0.0:
      idf_qwords_bigrams_in_doc_val = 0.0
    else:
      idf_qwords_bigrams_in_doc_val = (float(idf_qwords_bigrams_in_doc) /
                                       float(idf_bigrams))

    return [qwords_in_doc_val,
            qwords_bigrams_in_doc_val,
            idf_qwords_in_doc_val,
            idf_qwords_bigrams_in_doc_val]

  def get_words(self, s):
    sl = utils.bioclean(s)
    sl = [s for s in sl]
    return sl

  def GetExtraFeatures(self, qtext, dtext, bm25):
    qwords = self.get_words(qtext)
    dwords = self.get_words(dtext)
    qd1 = self.query_doc_overlap(qwords, dwords)
    bm25 = [bm25]
    return qd1[0:3] + bm25

  def Cosine(self, v1, v2):
    return dy.cdiv(dy.dot_product(v1, v2),
                   dy.l2_norm(v1) * dy.l2_norm(v2))

  def Conv(self, input_vecs):
    vecs_tri = []
    for tok in range(len(input_vecs)):
      ptok = (input_vecs[tok-1] if tok > 0 else self.pad[0])
      ntok = (input_vecs[tok+1] if tok < len(input_vecs)-1 else self.pad[1])
      ctok = input_vecs[tok]
      input_vec = dy.concatenate([ctok, ptok, ntok])
      cvec = utils.leaky_relu(self.W_conv.expr() * input_vec +
                              self.b_conv.expr())
      vecs_tri.append(cvec)

    conv_vecs = [dy.esum([iv, tv])
                 for iv, tv
                 in zip(input_vecs, vecs_tri)]
    return conv_vecs

  def MakeInputs(self, text):
    words = self.get_words(text)
    vecs = []
    wds = []
    for w in words:
      if w in self.wv:
        vec = dy.inputVector(self.wv[w])
        vecs.append(dy.nobackprop(vec))
        wds.append(w)
    conv = self.Conv(vecs)
    return wds, vecs, conv

  def GetQDScore(self, qwords, qreps, dwords, dreps, extra):
    nq = len(qreps)
    nd = len(dreps)
    qgl = [self.W_gate.expr() *
           dy.concatenate([qv, dy.constant(1, self.idf_val(qw))])
           for qv, qw in zip(qreps, qwords)]
    qgates = dy.softmax(dy.concatenate(qgl))

    qscores = []
    for qtok in range(len(qreps)):
      qrep = qreps[qtok]
      att_scores = [dy.dot_product(qrep, drep) for drep in dreps]
      att_probs = dy.softmax(dy.concatenate(att_scores))
      doc_rep = dy.esum([v * p for p, v in zip(att_probs, dreps)])
      input_vec = dy.cmult(qrep, doc_rep)
      #input_dot = dy.sum_elems(input_vec)
      #input_len = dy.l2_norm(qrep - doc_rep)
      #input_vec = dy.concatenate([input_vec, input_dot, input_len])

      layer = utils.leaky_relu(self.b_term.expr() +
                               self.W_term.expr() * input_vec)
      score = (self.b_term2.expr() +
               self.W_term2.expr() * layer)
      qscores.append(score)

    # Final scores and ultimate classifier.
    qterm_score = dy.dot_product(dy.concatenate(qscores), qgates)

    fin_score = (self.b_final.expr() +
                 self.W_final.expr() * dy.concatenate([qterm_score, extra]))
    return fin_score

  def Update(self, pos, neg):
    loss = []
    for p in pos:
      l = []
      l.append(p)
      for n in neg:
        l.append(n)
      loss.append(dy.hinge(dy.concatenate(l), 0))
    if len(loss) > 0:
      sum_loss = dy.esum(loss)
      sum_loss.scalar_value()
      sum_loss.backward()
      self.trainer.update()
    dy.renew_cg()

  def PairAppendToLoss(self, pos, neg, loss):
    if len(pos) != 1:
      print('ERROR IN POS EXAMPLE SIZE')
      print(len(pos))
    loss.append(dy.hinge(dy.concatenate(pos + neg), 0))

  def UpdateBatch(self, loss):
    if len(loss) > 0:
      sum_loss = dy.esum(loss)
      sum_loss.scalar_value()
      sum_loss.backward()
      self.trainer.update()
    dy.renew_cg()
