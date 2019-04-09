import os
import csv
from modularized_copynet import CopyNetSeq2Seq
import numpy as np
from IPython import embed
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import Dict, Tuple, List, Any, Union
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator, DataIterator
from allennlp.training.trainer import Trainer
from allennlp.data.tokenizers import Token
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.modules.attention.bilinear_attention import BilinearAttention
from allennlp.data import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL
import argparse
import random




class PerceptionDatasetReader(DatasetReader):
    
    def __init__(self, token_indexers=None):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
    def text_to_instance(self, in_tokens, out_tokens):
        in_sentence_field = TextField(in_tokens, self.token_indexers)
        out_sentence_field = TextField(out_tokens, self.token_indexers)
        fields = {"source_tokens": in_sentence_field, "target_tokens": out_sentence_field}

        return Instance(fields)
    
    def _read(self, file_path):
        with open(file_path) as f:
            csvreader = csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
            next(csvreader)
            for row in csvreader:
                yield self.text_to_instance([Token(START_SYMBOL)] + [Token(word.lower()) for word in word_tokenize(row[2].split('=')[1])] + [ Token(END_SYMBOL)], [Token(START_SYMBOL)] +  [Token(word.lower()) for word in word_tokenize(row[4])] + [Token(END_SYMBOL)])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_output_dim', type=int, default=50)
    parser.add_argument('--target_embedding_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--max_decoding_steps', type=int, default=30)
    parser.add_argument('--decoder_output_dim', type=int, default=50)

    args = parser.parse_args()

    encoder_output_dim = args.encoder_output_dim
    target_embedding_size = args.target_embedding_size
    batch_size = args.batch_size
    beam_size = args.beam_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    max_decoding_steps = args.max_decoding_steps
    decoder_output_dim = args.decoder_output_dim


    '''
    Loading Dataset
    '''
    datasetreader= PerceptionDatasetReader()

    '''
    train_dataset & test_dataset are a list of Instances
    Each Instance contains two fields (instance.fields) as keys: 'source_tokens' and 'target_tokens'
    instance.fields['source_tokens'].tokens is a list of Token objects
    '''


    train_dataset = datasetreader.read('generation_train.tsv')
    test_dataset = datasetreader.read('generation_test.tsv')

    # cut = int(0.2*len(train_dataset))
    # validation_dataset = train_dataset[:cut]
    # train_dataset = train_dataset[cut:]

    '''
    Creating the vocabulary from instances
    vocab.get_token_from_index(index), where index is an int
    vocab.get_token_index(token), where token is a string
    vocab.get_index_to_token_vocabulary()
    vocab.get_token_to_index_vocabulary()
    '''

    vocab = Vocabulary.from_instances(train_dataset + test_dataset)

    '''
    Loading the embeddings
    '''
    
    embedding_dim = 50

    '''
    Params is a wrapper for parameters
    '''
    embedding_params = Params({'pretrained_file': 'new_embeddings.txt', 'vocab_namespace': 'tokens','embedding_dim': embedding_dim})


    '''
    token_embedding.parameters() returns a generator of the embeddings. 
    list(token_embedding.parameters())[0] is a 2-D tensor object
    list(token_embedding.parameters())[0][index] is a vector of length 50 
    '''
    token_embedding = Embedding.from_params(vocab,embedding_params)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    '''
    Encoder LSTM wrapped by a Seq2Seq Wrapper. 
    '''
    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(embedding_dim, encoder_output_dim, batch_first=True, bidirectional=True))
    attention = BilinearAttention(vector_dim = decoder_output_dim, matrix_dim = 2*encoder_output_dim, normalize=False)


    model = CopyNetSeq2Seq(vocab, word_embeddings, encoder=encoder, attention= attention, target_embedding_dim = target_embedding_size, max_decoding_steps=max_decoding_steps, beam_size=beam_size,decoder_output_dim = decoder_output_dim)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("source_tokens", "num_tokens"), ("target_tokens", "num_tokens")], padding_noise=0.5)
    # iterator = DataIterator(batch_size=batch_size)
    iterator.index_with(vocab)


    trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=test_dataset,
                  patience=3,
                  num_epochs=num_epochs)
    trainer.train()
    


    embed()


if __name__ == "__main__":
    main()
