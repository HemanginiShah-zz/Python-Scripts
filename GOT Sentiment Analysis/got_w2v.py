#How to Make Word Vectors from Game of Throne
# --------
# Tutorial from Siraj Raval
#   -> https://youtu.be/pY9EwZ02sXU?list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3
# # #
# Developed by Nathan Shepherd

print('Loading dependencies and fetching corpus data from file ...')
from nltk import word_tokenize, sent_tokenize, punkt
from nltk.corpus import  stopwords
import tensorflow as tf

#for word encoding
import codecs
#regex
import glob
#concurrency
import multiprocessing
#dealing with os
import os
#pretty printing
import pprint
#regular expressions
import re
#google trained word vectors
import word2vec as w2v
#demensionality reduction
import sklearn.manifold
#math
import numpy
#plotting
import matplotlib.pyplot as plt
#parse pandas as pd
import pandas as pd
#visualization
import seaborn as sns


#get book names, matching txt file
corpus_raw = u""
b1 = open('got1.txt').read()
b2 = open('got2.txt').read()
b3 = open('got3.txt').read()
b4 = open('got4.txt').read()
b5 = open('got5.txt').read()
corpus_raw += b1+b2+b3+b4+b5

print('Corpus is now {0} characters long\n'.format(len(corpus_raw)))

#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#raw_sentences = tokenizer.tokenize(corpus_raw)

print("Tokenizing corpus, please wait ...")
raw_sentences = sent_tokenize(corpus_raw)

def sent_to_wordlist(ins):
    clean = re.sub("[^a-zA-Z0-9]", " ", ins)
    words = word_tokenize(clean)
    return words

sentences = []
for sent in raw_sentences:
    if len(sent) > 0:
        sentences.append(sent_to_wordlist(sent))

print('Successfully intialized sentence matrix of word arrays')

## Word2Vec
# Build Model
#dimensionality of resulting word vectors
num_features = 300
#minimum word count threshold
min_word_count = 3
#number of threads to run in parallel
num_workers = multiprocessing.cpu_count()
#context window length
context_size = 7
#downsample setting frequent qords
downsampling = '1e-3'
#seed for the RNG, to make results reproducable
seed = 1

thrones2vec = w2v.word2vec(
    train=sentences,
    output="thrones2vec.w2v",
    #threads=4,
    #size=num_features,
    #min_count=min_word_count,
    #window=context_size,
    )
thrones2vec.build_vocab(sentences)
print("Word2Vec vocabulary length:", len(thrones2vec.vocab))

## Start Training
print('\n Training word vectors, this may take a few minutes ...')
thrones2vec.train(sentences)

if not os.path.exists('Trained GOT Vectors'):
    os.makedirs('Trained GOT Vectors')
thrones2vec.save(os.path.join('Trained GOT Vectors', "thrones2vec.w2v"))

