from gensim import corpora, models


def gen_tfidf_model(corpus_path, dst):
    corpus = corpora.MmCorpus(corpus_path)
    tfidf = models.TfidfModel(corpus)
    tfidf.save(dst)


if __name__ == '__main__':
    gen_tfidf_model('data/train_corpus.mm', 'data/train.tfidf_model')
