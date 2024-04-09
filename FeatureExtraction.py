import os

import numpy as np
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import FastText, Doc2Vec

vectorizer_tfidf = TfidfVectorizer()
vectorizer_cv = CountVectorizer()


#TFIDF  : data read as .values
def vectorizeTfidf(x_train,x_test,x_valid):
    x_train = vectorizer_tfidf.fit_transform(x_train)
    x_test = vectorizer_tfidf.transform(x_test)
    x_valid = vectorizer_tfidf.transform(x_valid)

    x_train.sort_indices()
    x_test.sort_indices()
    x_valid.sort_indices()
    return x_train,x_test,x_valid
def vectorizeTfidf_Transform(review):
    review = vectorizer_tfidf.transform(review)
    review.sort_indices()
    return review


#COUNT VECTORIZER : data read as .values
def vectorizeCV(x_train,x_test,x_valid):
    x_train = vectorizer_cv.fit_transform(x_train)
    x_test = vectorizer_cv.transform(x_test)
    x_valid = vectorizer_cv.transform(x_valid)

    x_train.sort_indices()
    x_test.sort_indices()
    x_valid.sort_indices()
    return x_train, x_test, x_valid
def vectorizeCV_Transform(review):
    review = vectorizer_cv.transform(review)
    review.sort_indices()
    return review




#DOV2VEC
def vectorizeDoc2Vec(x_train, x_test, x_valid):
    if os.path.exists("Vectorization_Model_Doc2Vec.model"):
        print("Doc2Vec model is already exists. Returning the vectors...")
        model = Doc2Vec.load("Vectorization_Model_Doc2Vec.model")
        # Get vectors for all data
        x_train_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_train]
        x_test_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_test]
        x_valid_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_valid]

        return x_train_vectors, x_test_vectors, x_valid_vectors
    else:
        # Tokenize the training data
        tagged_train = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(x_train)]
        # Initialize the model
        model = Doc2Vec(vector_size=300, window=2, min_count=1, workers=4, epochs=10)
        # Build the vocabulary
        model.build_vocab(tagged_train)
        # Train the model
        model.train(tagged_train, total_examples=model.corpus_count, epochs=model.epochs)
        # Save the model
        model.save("Vectorization_Model_Doc2Vec.model")

        # Get vectors for all data
        x_train_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_train]
        x_test_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_test]
        x_valid_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_valid]

        return x_train_vectors, x_test_vectors, x_valid_vectors
def vectorizeDoc2Vec_Transform(review):
    if os.path.exists("Vectorization_Model_Doc2Vec.model"):
        model = Doc2Vec.load("Vectorization_Model_Doc2Vec.model")
        vector = model.infer_vector(review)
        return vector
    else:
        print("Doc2Vec model is already not exists. You need to run vectorizeDoc2Vec function first and you have to give 3 parameters to function.")
        return review




#FASTTEXT
def vectorizeFastText(x_train,x_test,x_valid):
    if os.path.exists("Vectorization_Model_FastText.model"):
        print("FastText model is already exists. Returning the vectors...")
        model = FastText.load("Vectorization_Model_FastText.model")
        tokenized_train = [sentence.split() for sentence in x_train]
        # Get vectors for all data
        x_train_vectors = [model.wv.get_vector(' '.join(sentence)) for sentence in tokenized_train]
        x_test_vectors = [model.wv.get_vector(' '.join(sentence.split())) for sentence in x_test]
        x_valid_vectors = [model.wv.get_vector(' '.join(sentence.split())) for sentence in x_valid]
        return x_train_vectors, x_test_vectors, x_valid_vectors
    else:
        # Tokenize the training data
        tokenized_train = [sentence.split() for sentence in x_train]
        # Initialize the model
        model = FastText(vector_size=300, window=2, min_count=1, workers=4)
        model.build_vocab(tokenized_train, update=False)
        # Train the model
        model.train(tokenized_train, total_examples=len(tokenized_train), epochs=model.epochs)
        # Save the model
        model.save("Vectorization_Model_FastText.model")
        # Get vectors for all data
        x_train_vectors = [model.wv.get_vector(' '.join(sentence)) for sentence in tokenized_train]
        x_test_vectors = [model.wv.get_vector(' '.join(sentence.split())) for sentence in x_test]
        x_valid_vectors = [model.wv.get_vector(' '.join(sentence.split())) for sentence in x_valid]

        return x_train_vectors, x_test_vectors, x_valid_vectors
def vectorizeFastText_Transform(review):
    # Load the FastText model
    model = FastText.load("Vectorization_Model_FastText.model")
    # Tokenize the review
    tokenized_review = review[0].split()
    # Get the vector for the review
    review_vector = model.wv.get_vector(' '.join(tokenized_review))
    #review_vector = model.wv.get_vector(' '.join(review[0].split()))

    return review_vector



if os.path.exists("D:/NLP_Models/cc.tr.300.bin"):
    model_path = "D:/NLP_Models/cc.tr.300.bin"
    model = FastText.load_fasttext_format(model_path)
def vectorize_FastText_Pretrained(x_train, x_test, x_valid):

    # Vectorize x_train
    x_train_vectors = []
    for text in x_train:
        vector = model.wv[text]
        x_train_vectors.append(vector)

    # Vectorize x_test
    x_test_vectors = []
    for text in x_test:
        vector = model.wv[text]
        x_test_vectors.append(vector)

    # Vectorize x_valid
    x_valid_vectors = []
    for text in x_valid:
        vector = model.wv[text]
        x_valid_vectors.append(vector)

    return x_train_vectors, x_test_vectors, x_valid_vectors
def vectorize_FastText_Pretrained_Transform(review):

    # Vektörünü almak istediğiniz cümle
    sentence = review[0]
    # Cümleyi vektöre dönüştür
    vector = model.wv[sentence]
    vector = np.reshape(vector, (1, -1))
    return vector



#OLD FUNCTION
"""""
def vectorizeDoc2Vec(x_train, x_test, x_valid):
    if os.path.exists("Vectorization_Model_Doc2Vec.model"):
        print("Doc2Vec model is already exists. You can run vectorizeDoc2Vec_Transform function, you have to give one 'review' parameter.")
        return x_train,x_test,x_valid
    else:
        # Tokenize the training data
        tagged_train = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(x_train)]

        # Initialize the model
        model = Doc2Vec(vector_size=300, window=2, min_count=1, workers=4, epochs=10)

        # Build the vocabulary
        model.build_vocab(tagged_train)

        # Train the model
        model.train(tagged_train, total_examples=model.corpus_count, epochs=model.epochs)

        # Get vectors for all data
        x_train_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_train]
        x_test_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_test]
        x_valid_vectors = [model.infer_vector(word_tokenize(_d.lower())) for _d in x_valid]

        model.save("Vectorization_Model_Doc2Vec.model")
        return x_train_vectors, x_test_vectors, x_valid_vectors
"""""