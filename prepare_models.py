import itertools

import scipy as np
from gensim import corpora, models
from gensim.models.doc2vec import *
from nltk.corpus import stopwords
from six import iteritems

from read_data import readobj
from settings import *
from time import time


class FTData:
    stopwds = set(stopwords.words('english'))

    def __init__(self, path):
        self.gener = readobj(path, 0)

    def __iter__(self):
        for em in self.gener:
            yield [
                [w for w in filter(lambda w: w not in self.stopwds, st.lower().split())]
                for st in em.body
            ]


def gendict(pbpath, dst):
    docs = itertools.chain.from_iterable(FTData(pbpath))
    # print(docs)
    dictionary = corpora.Dictionary(docs)
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    dictionary.save(dst)


def gencorpus(pbpath, dictpath, dst):
    dictionary = corpora.Dictionary.load(dictpath)
    corpus = [
        dictionary.doc2bow(w) for s in FTData(pbpath) for w in s
    ]
    # pprint(corpus)
    corpora.MmCorpus.serialize(dst, corpus)


def gen_tfidf_model(corpus_path, dst):
    corpus = corpora.MmCorpus(corpus_path)
    tfidf = models.TfidfModel(corpus)
    # print(tfidf[[(281, 1), (591, 1), (1601, 1)]])
    tfidf.save(dst)


class Embedding(object):
    stopwds = set(stopwords.words('english'))

    def __init__(self, path):
        self.gener = readobj(path, 0)
        self.model = Doc2Vec(min_count=1, window=10, size=400, alpha=0.025, min_alpha=0.001, sample=1e-3, negative=5,
                             workers=24)
        self.num_examples = 240000

    @staticmethod
    def clean_doc(doc):
        clean = []
        for st in doc:
            clean.extend(w for w in st.lower().split() if w not in Embedding.stopwds)
        return clean

    @staticmethod
    def getLabeledSentence(gener):
        for em in gener:
            yield TaggedDocument(Embedding.clean_doc(em.body), [str(int(em.messageid))])

    @staticmethod
    def getVecs(model, corpus, size):
        return np.concatenate(np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus)

    @staticmethod
    def getVec(model, doc, size):
        return np.array(model.docvecs[doc.tags[0]]).reshape((1, size))

    def train(self, epoch_num=50, save_path=None):
        sentences = self.getLabeledSentence(self.gener)
        # 使用所有的数据建立词典
        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=self.num_examples, epochs=epoch_num)
        if save_path:
            self.model.save(save_path)
        return self.model


if __name__ == '__main__':
    settings = SETTINGS['train']
    model_settings = settings['model']
    gendict(settings['data'], model_settings['dict'])
    gencorpus(settings['data'], model_settings['dict'], model_settings['corpus'])
    gen_tfidf_model(model_settings['corpus'], model_settings['tfidf'])

    emb = Embedding(settings['data'])
    t1 = time()
    emb.train(save_path=model_settings['wordvec'])
    print("Time used = %d\n" % (time() - t1))
    # model = Word2Vec.load(model_settings['wordvec'])
    # vecs = getVecs(model, Embedding.getLabeledSentence(readobj(settings['data'])), 400)
    # print(vecs[0:5])
