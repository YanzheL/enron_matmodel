from read_data import readobj
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from collections import defaultdict


class DataSet(object):
    stopwds = set(stopwords.words('english'))
    _num_examples = 240000

    def __init__(self):
        self.gener = readobj('data/train.pb', 0)

    @property
    def num_examples(self):
        return self._num_examples

    def process_sentence(self, sentences):
        texts = [
            [word for word in sentence.lower().split() if word not in self.stopwds]
            for sentence in sentences
        ]

        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [
            [token for token in text if frequency[token] > 1]
            for text in texts
        ]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

    def filter_stopwds(self, s):
        return ''.join([w for w in s.split().lower() if w not in self.stopwds])

    def next_batch(self, batch_size):
        ems = []
        try:
            for i in range(batch_size):
                em = self.gener.__next__()
        except StopIteration:
            pass

        pass
