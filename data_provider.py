import scipy as np
from gensim.models.doc2vec import *
from nltk.corpus import stopwords

from inference import OUTPUT_NODE, INPUT_NODE
from prepare_models import Embedding
from read_data import readobj
from settings import *


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
    def vecmodel(self):
        return self._vecmodel

    def next_batch(self, batch_size):

        xs = []
        ys = []
        for i, lbs in enumerate(Embedding.getLabeledSentence(self.gener)):
            if i >= batch_size:
                break
            label = np.zeros((1, OUTPUT_NODE), dtype=float)
            x = Embedding.getVec(self.vecmodel, lbs, INPUT_NODE)
            id = lbs.tags[0]
            print(id)
            if int(id) < OUTPUT_NODE:
                label[0, int(id)] = 1.
            xs.append(x)
            ys.append(label)

        return xs, ys
