# AUEB at BioASQ 6: Document and Snippet Retrieval

This software accompanies  the following paper:
>G. Brokos, P. Liosis, R. McDonald, D. Pappas and I. Androutsopoulos, "AUEB at BioASQ 6: Document and Snippet Retrieval". Proceedings of the workshop BioASQ: Large-scale Biomedical Semantic Indexing and Question Answering, at the Conference on Empirical Methods in Natural Language Processing (EMNLP 2018), Brussels, Belgium, 2018. [[PDF](http://nlp.cs.aueb.gr/pubs/aueb_at_bioasq6.pdf)]

# Instructions
This is a Python 3.6 project.

**Step 1**: Install the required Python packages: 

```
pip3 install -r requirements.txt
```

**Step 2**: Download the necessary data that will be used as input to the models.

```
sh get_bioasq6_data.sh
```

The following data are provided (among other files):

* Top-k documents retrieved by a BM25 based search engine ([Galago](http://www.lemurproject.org/galago.php)) for each BioASQ query.
* Biomedical pre-trained word embeddings
* IDF values

*Note: Downloading time may vary depending on server availability.*

**Step 3**: Navigate to a models directory to train the specific model and evaluate its performance on each one of the five test batches. E.g. navigate to the TERM-PACRR model for document ranking:
```
cd models/documents/term-pacrr
```
Consult the README file of each model for dedicated instructions (e.g. [instructions for TERM-PACRR](https://github.com/nlpaueb/aueb-bioasq6/tree/master/models/documents/term-pacrr#term-pacrr)).

(Under construction.)
