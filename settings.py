SETTINGS = {
    'train': {
        'data': 'data/train.pb',
        'model': {
            'dict': 'model/train.dict',
            'corpus': 'model/train_corpus.mm',
            'tfidf': 'model/train_tfidf.model',
            'wordvec': 'model/train_word2vec.model'
        }
    },
    'validate': {
        'data': 'data/validate.pb',
        'model': {
            'dict': 'model/validate.dict',
            'corpus': 'model/validate_corpus.mm',
            'tfidf': 'model/validate_tfidf.model',
            'wordvec': 'model/validate_word2vec.model'
        }
    }
}
