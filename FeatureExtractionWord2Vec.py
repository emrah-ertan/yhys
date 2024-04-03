import os.path

import nltk
from gensim.models import Word2Vec


model_word2vec = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)


def vectorizeWord2Vec(x_train,x_test):
    train_sentences = [nltk.word_tokenize(sentence) for sentence in x_train]
    test_sentences = [nltk.word_tokenize(sentence) for sentence in x_test]

    model_word2vec.build_vocab(train_sentences)
    model_word2vec.train(train_sentences, total_examples=model_word2vec.corpus_count, epochs=10)

    train_vectors = [model_word2vec.wv[sentence] for sentence in train_sentences]

    model_word2vec.save("models/word2vecModel.model")

    test_vectors = [model_word2vec.wv[sentence] for sentence in test_sentences]

    return train_vectors, test_vectors

def vectorizeWord2Vec_Transform(review):
    if os.path.exists("models/word2vecModel.model"):
        review_vector = model_word2vec.wv[nltk.word_tokenize(review)]
        return review_vector
    else:
        print(f"Word2Vec Modeli Bulunamadı. vectorizeWord2Vec() ile oluşturmayı deneyin")