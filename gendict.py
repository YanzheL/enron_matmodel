from read_data import readobj
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from collections import defaultdict
from six import iteritems


class IterCorpus:
    stopwds = set(stopwords.words('english'))

    def __init__(self):
        self.gener = readobj('data/train.pb', 0)

    def __iter__(self):
        for em in self.gener:
            for st in em.body:
                yield filter(lambda w: w not in self.stopwds, st.lower().split())
                # yield self.dictionary.doc2bow(filter(lambda w: w not in self.stopwds, st.lower().split()))
                # print(st)
                # yield 'sdf sda re w'.lower().split()
                # yield self.dictionary.doc2bow(filter(lambda w: w not in self.stopwds, st.lower().split()))

                # def rm_invalid(self):
                #     once_ids = [tokenid for tokenid, docfreq in iteritems(self.dictionary.dfs) if docfreq == 1]
                #     self.dictionary.filter_tokens(once_ids)


if __name__ == '__main__':
    corpus = IterCorpus()

    dictionary = corpora.Dictionary(corpus)

    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(once_ids)

    # for i in dictionary:
    #     print(i)

    dictionary.save('data/train.dict')

    # dictionary = corpora.Dictionary()
    # once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    # dictionary.filter_tokens(once_ids)

    # corpora.MmCorpus.serialize('data/validate_corpus.mm', vec)
    #
    # print(vec)
