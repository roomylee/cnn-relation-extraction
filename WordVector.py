import gensim
import os
import pandas as pd
import nltk


class WordVector:
    def __init__(self, dim=100):
        # train_corpus is not None -> Train Again
        self.dim = dim
        self.corpus = self.build_corpus()
        self.model = self.build_model()

    def build_corpus(self):
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')

        corpus = []
        for s in train['sentence']:
            tokens = nltk.word_tokenize(s)
            corpus.append(tokens)
        for s in test['sentence']:
            tokens = nltk.word_tokenize(s)
            corpus.append(tokens)
        return corpus

    def build_model(self):
        try:
            print("Loading a pre-trained model...")
            model = gensim.models.Word2Vec.load(os.path.dirname(__file__)+"/wv_model/word2vec_%d_dim.model" % self.dim)
            print("Load success!")
        except Exception:
            print("Training a word2vec model...")
            model = self.train(self.corpus)
            print("Training success!")

        return model

    def train(self, corpus):
        print('Train Data Size :', len(corpus))
        model = gensim.models.Word2Vec(corpus, min_count=1, size=self.dim)
        model.save(os.path.dirname(__file__)+"/wv_model/word2vec_%d_dim.model" % self.dim)

        return model


if __name__ == '__main__':
    # word = ['happy']
    # corpus = [['I','am','happy']]
    # wv = WordVector(word, corpus)
    wv = WordVector(dim=3)
