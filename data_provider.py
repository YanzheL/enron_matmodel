from read_data import readobj
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from collections import defaultdict

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

    def __init__(self, datasource):
        self.gener = readobj(datasource, 0)

    @property
    def num_examples(self):
        return self._num_examples


    def next_batch(self, batch_size):
        ems = []
        try:
            for i in range(batch_size):
                em = self.gener.__next__()
        except StopIteration:
            pass

        pass
