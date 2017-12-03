from read_data import readobj
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from collections import defaultdict
from gensim.models.doc2vec import *
from settings import *
from prepare_models import Embedding
import scipy as np
from inference import OUTPUT_NODE, INPUT_NODE


class FilteredData:
    stopwds = set(stopwords.words('english'))

    def __init__(self, path):
        self.gener = readobj(path, 0)

    def __iter__(self):
        for em in self.gener:
            yield [
                [w for w in filter(lambda w: w not in self.stopwds, st.lower().split())]
                for st in em.body
            ]


class DataSet(object):
    stopwds = set(stopwords.words('english'))
    _num_examples = 240000
    _vecmodel = Word2Vec.load(SETTINGS['train']['model']['wordvec'])

    def __init__(self, datasource=SETTINGS['train']['data']):
        self.gener = readobj(datasource)

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def vecmodelel(self):
        return self._vecmodel

    def next_batch(self, batch_size):

        xs = []
        ys = []
        for i, lbs in enumerate(Embedding.getLabeledSentence(self.gener)):
            if i >= batch_size:
                break
            label = np.zeros((1, OUTPUT_NODE), dtype=float)
            x = Embedding.getVec(self.vecmodelel, lbs, INPUT_NODE)
            id = lbs.tags[0]
            if id < OUTPUT_NODE:
                label[id] = 1.
            xs.append(x)
            ys.append(label)

        return xs,ys
