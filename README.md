This is an AllenNLP implementation of the model used in the paper [How Much is 131 Million Dollars? Putting Numbers in Perspective with Compositional Descriptions](https://www.aclweb.org/anthology/P16-1055)

The data can be downloaded from the [CodaLab Worksheet](https://worksheets.codalab.org/worksheets/0x243284b4d81d4590b46030cdd3b72633/). Specifically, you would need `generation_train.tsv` and `generation_test.tsv`

The code expects pre-trained word-embedding file as a txt file with each line containing space separated tokens and the vector embeddings. e.g. for 3 dimensional word vectors, the file would look something like this:

`the 0.323 0.23542 1.324
and 0.2342 2.235 1..1352`

To train the model, you can run `train.sh`. You can also modify the different parameters given in `train.sh`. To obtain different BLEU scores, uncomment the corresponding line of code in the function `get_metrics` in `seq2seq_copy.py`. 
