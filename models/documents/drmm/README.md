# ABEL-DRMM
Run DRMM ABEL model on the BioASQ data:
```
python3 drmm_main.py --dynet-mem 2000
```
Models and output predictions on the dev set are dumped after every iteration over the data.

Let's say on epoch 5 you have the best dev accuracy. This can be tested by going into the top-level eval/ directory.

To get the test predictions for a batch, say batch 1, run:
```
python3 drmm_predict.py abel_model_ep5 1 --dynet-mem 2000
```
These are stored in abel_test_preds_batch1.json
