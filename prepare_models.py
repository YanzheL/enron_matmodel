from read_data import readobj
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from collections import defaultdict
from six import iteritems
from gensim import corpora, models
from pprint import pprint
import itertools


class IterCorpus:
    stopwds = set(stopwords.words('english'))

    def __init__(self, path):
        self.gener = readobj(path, 0)

    def __iter__(self):
        for em in self.gener:
            yield [
                [w for w in filter(lambda w: w not in self.stopwds, st.lower().split())]
                for st in em.body
            ]
            #
            # for st in em.body:
            #     yield filter(lambda w: w not in self.stopwds, st.lower().split())


def gendict(pbpath, dst):
    docs = itertools.chain.from_iterable(IterCorpus(pbpath))
    # print(docs)
    dictionary = corpora.Dictionary(docs)
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    dictionary.save(dst)


def gencorpus(pbpath, dictpath, dst):
    dictionary = corpora.Dictionary.load(dictpath)
    corpus = [
        dictionary.doc2bow(w) for s in IterCorpus(pbpath) for w in s
    ]
    # pprint(corpus)
    corpora.MmCorpus.serialize(dst, corpus)


def gen_tfidf_model(corpus_path, dst):
    corpus = corpora.MmCorpus(corpus_path)
    tfidf = models.TfidfModel(corpus)
    # print(tfidf[[(281, 1), (591, 1), (1601, 1)]])
    tfidf.save(dst)


if __name__ == '__main__':
    train_data_path = 'data/train.pb'
    # train_data_path = 'data/validate.pb'
    train_dict_path = 'data/train.dict'
    # train_dict_path = 'data/validate.dict'
    train_corpus_path = 'data/train_corpus.mm'
    # train_corpus_path = 'data/validate_corpus.mm'
    tfidf_model_path = 'data/train.tfidf_model'
    # tfidf_model_path = 'data/validate.tfidf_model'

    gendict(train_data_path, train_dict_path)
    gencorpus(train_data_path, train_dict_path, train_corpus_path)
    gen_tfidf_model(train_corpus_path, tfidf_model_path)

    # gencorpus(train_data_path, train_dict_path, train_corpus_path)
