# TERM-PACRR

Run TERM-PACRR:
```

```
This script will train TERM-PACRR using the tuned (on development) hyperparameters. The best state of the model (epoch with best performance on the dev) will then be used to rerank the top 100 documents retrieved with BM25, for the queries of each one of the five BioASQ test batches.
