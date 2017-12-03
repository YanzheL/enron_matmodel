from gendict import IterCorpus
from read_data import readobj
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from collections import defaultdict
from six import iteritems

from pprint import pprint

if __name__ == '__main__':
    dictionary = corpora.Dictionary.load('data/train.dict')
    itr = IterCorpus('data/train.pb')
    corpus = [dictionary.doc2bow(w) for w in itr]
    corpora.MmCorpus.serialize('data/train_corpus.mm', corpus)
    # pprint(corpus)
