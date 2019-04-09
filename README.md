To train the model, you can run train.sh. You can also modify the different parameters given in train.sh. To obtain different BLEU scores, uncomment the corresponding line of code in the function get_metrics in modularized_copynet.py. 

Further, if you want to run copynet, run "python train_copynet.py".

The training and the test data need to be in tab separated files. However, the embeddings file should be a txt file with each line containing space separated tokens and the vector embeddings. 
e.g. for a 3 dimension vector, it looks like this:
the 0.323 0.23542 1.324
and 0.2342 2.235 1..1352
