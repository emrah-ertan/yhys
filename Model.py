import pickle
import os

import numpy as np
from keras import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import Visualize


#logistic regression model for tfidif and cv vectors
def lrModel(x_train,y_train,x_test,y_test):
    if os.path.exists("models/LogisticRegressionModel.pkl"):
        #print("Model dosyası zaten mevcut. lrPredict() fonksiyonunu kullanabilirsiniz.")
        pass
    else:
        model = LogisticRegression()    #random_state parametresi doğruluk üzerinde etkili olabilir
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)

        cm = confusion_matrix(y_test,y_pred)
        print("Confusion Matrix:")
        print(cm)
        Visualize.showConfusionMatrix(cm)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of Model:", accuracy)

        #Model kaydedilir
        with open('models/LogisticRegressionModel.pkl', 'wb') as f:
            pickle.dump(model, f)
def lrPredict(review_vector):
    with open('models/LogisticRegressionModel.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(review_vector)

    if prediction == 1:
        print(f"Metin Sınıfı 1: Saldırganlık, zorbalık veya linç tespit edildi.")
    else:
        print("Metin Sınıfı 0: Saldırganlık, zorbalık veya linç tespit edilemedi.")




#Naive Bayes
def nbModel(x_train, y_train, x_test, y_test):
    if os.path.exists("models/NaiveBayesModel.pkl"):
        #print("Model dosyası zaten mevcut. nbPredict() fonksiyonunu kullanabilirsiniz.")
        pass
    else:
        model = MultinomialNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        Visualize.showConfusionMatrix(cm)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of Model:", accuracy)

        with open('models/NaiveBayesModel.pkl', 'wb') as f:
            pickle.dump(model, f)
def nbPredict(review_vector):
    with open('models/NaiveBayesModel.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(review_vector)
    if prediction == 1:
        print(f"Metin Sınıfı 1: Saldırganlık, zorbalık veya linç tespit edildi.")
    else:
        print("Metin Sınıfı 0: Saldırganlık, zorbalık veya linç tespit edilemedi.")





#Random Forest
def rfModel(x_train, y_train, x_test, y_test):
    if os.path.exists("models/RandomForestModel.pkl"):
        #print("Model dosyası zaten mevcut. rfPredict() fonksiyonunu kullanabilirsiniz.")
        pass
    else:
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        Visualize.showConfusionMatrix(cm)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of Model:", accuracy)

        # Model kaydedilir
        with open('models/RandomForestModel.pkl', 'wb') as f:
            pickle.dump(model, f)
def rfPredict(review_vector):
    with open('models/RandomForestModel.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(review_vector)

    if prediction == 1:
        print(f"Metin Sınıfı 1: Saldırganlık, zorbalık veya linç tespit edildi.")
    else:
        print("Metin Sınıfı 0: Saldırganlık, zorbalık veya linç tespit edilemedi.")





#Support Vector Machine
def svmModel(x_train, y_train, x_test, y_test):
    if os.path.exists("models/SVMModel.pkl"):
        #print("Model dosyası zaten mevcut. svmPredict() fonksiyonunu kullanabilirsiniz.")
        pass
    else:
        model = SVC()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        Visualize.showConfusionMatrix(cm)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of Model:", accuracy)

        # Model kaydedilir
        with open('models/SVMModel.pkl', 'wb') as f:
            pickle.dump(model, f)

def svmPredict(review_vector):
    with open('models/SVMModel.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(review_vector)

    if prediction == 1:
        print(f"Metin Sınıfı 1: Saldırganlık, zorbalık veya linç tespit edildi.")
    else:
        print("Metin Sınıfı 0: Saldırganlık, zorbalık veya linç tespit edilemedi.")




#Decision Tree
def dtModel(x_train, y_train, x_test, y_test):
    if os.path.exists("models/DecisionTreeModel.pkl"):
        #print("Model dosyası zaten mevcut. dtPredict() fonksiyonunu kullanabilirsiniz.")
        pass
    else:
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        Visualize.showConfusionMatrix(cm)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of Model:", accuracy)

        # Model kaydedilir
        with open('models/DecisionTreeModel.pkl', 'wb') as f:
            pickle.dump(model, f)

def dtPredict(review_vector):
    with open('models/DecisionTreeModel.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(review_vector)

    if prediction == 1:
        print(f"Metin Sınıfı 1: Saldırganlık, zorbalık veya linç tespit edildi.")
    else:
        print("Metin Sınıfı 0: Saldırganlık, zorbalık veya linç tespit edilemedi.")






#Deep learning model for tfidf and cv vectors. Write the name of the vectorization method at the end of the model name
def dlModel(x_train,y_train,x_test,y_test,x_valid,y_valid):     # This function be able to use with tfidf and count vectorizer vectors
    if os.path.exists("models/DeepLearningModel_1.keras"):
        #print(f"Model dosyası zaten mevcut. dlPredict() fonksiyonunu kullanabilirsiniz.")
        pass
    else:
        model = Sequential()
        # model.add(Dense(7, activation='relu', input_shape=(x_train.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Early stopping için callback oluştur
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_valid, y_valid),callbacks=[early_stopping], initial_epoch=0)
        # Modelin performansını değerlendirme
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Test veri kümesi üzerinde kayıp (loss): {loss}")
        print(f"Test veri kümesi üzerinde doğruluk (accuracy): {accuracy}")
        y_pred = model.predict(x_test)
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
        cm = confusion_matrix(y_test,y_pred)
        print(f"Confusion Matrix: ")
        print(cm)
        Visualize.showConfusionMatrix(cm)
        model.save("models/DeepLearningModel_1.keras")
def dlPredict(review_vector):
    try:
        model = load_model("models/DeepLearningModel_1.keras")
        prediction = model.predict(review_vector)
        prediction = 1 if prediction >= 0.5 else 0
        if prediction == 1:
            print(f"Metin Sınıfı 1: Saldırganlık, zorbalık veya linç tespit edildi.")
        else:
            print("Metin Sınıfı 0: Saldırganlık, zorbalık veya linç tespit edilemedi.")
    except FileNotFoundError:
        print(f"İlgili model dosyası bulunamadı. Lütfen dlModel() fonksiyonunu kullandığınızdan ve dosya varlığından emin olun.")






#deep learning model for fasttext and doc2vec vectors
def dlModel_2(x_train,y_train,x_test,y_test,x_valid,y_valid):   # This function be able to use with doc2vec and fasttext vectors
    if os.path.exists("models/DeepLearningModel_2.keras"):
        #print(f"Model dosyası zaten mevcut. dlPredict() fonksiyonunu kullanabilirsiniz.")
        pass
    else:
        model = Sequential()
        #model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Early stopping için callback oluştur
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid), callbacks=[early_stopping], initial_epoch=0)
        # Modelin performansını değerlendirme
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Test veri kümesi üzerinde kayıp (loss): {loss}")
        print(f"Test veri kümesi üzerinde doğruluk (accuracy): {accuracy}")
        y_pred = model.predict(x_test)
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
        cm = confusion_matrix(y_test,y_pred)
        print(f"Confusion Matrix: ")
        print(cm)
        Visualize.showConfusionMatrix(cm)
        model.save("models/DeepLearningModel_2.keras")
def dlPredict_2(review_vector):
    try:
        model = load_model("models/DeepLearningModel_2.keras")
        prediction = model.predict(review_vector)
        prediction = 1 if prediction >= 0.5 else 0
        if prediction == 1:
            print(f"Metin Sınıfı 1: Saldırganlık, zorbalık veya linç tespit edildi.")
        else:
            print("Metin Sınıfı 0: Saldırganlık, zorbalık veya linç tespit edilemedi.")
    except FileNotFoundError:
        print(f"İlgili model dosyası bulunamadı. Lütfen dlModel_2() fonksiyonunu kullandığınızdan ve dosya varlığından emin olun.")




#LSTM architecture
def dlModel_LSTM(x_train,y_train,x_test,y_test,x_valid,y_valid):
    if os.path.exists("models/DeepLearningModel_LSTM.keras"):
        # print(f"Model dosyası zaten mevcut. dlPredict() fonksiyonunu kullanabilirsiniz.")
        pass
    else:

        x_train = x_train.toarray()
        x_test = x_test.toarray()
        x_valid = x_valid.toarray()

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_valid = np.reshape(x_valid, (x_valid.shape[0], 1, x_valid.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        model = Sequential()
        model.add(LSTM(units=32,activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_valid, y_valid), callbacks=[early_stopping])

        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Test veri kümesi üzerinde kayıp (loss): {loss}")
        print(f"Test veri kümesi üzerinde doğruluk (accuracy): {accuracy}")
        y_pred = model.predict(x_test)
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix: ")
        print(cm)
        Visualize.showConfusionMatrix(cm)
        model.save("models/DeepLearningModel_LSTM.keras")
def dlPredict_LSTM(review_vector):
    try:
        review_vector = review_vector.toarray()
        review_vector = np.reshape(review_vector, (review_vector.shape[0], 1, review_vector.shape[1]))
        model = load_model("models/DeepLearningModel_LSTM.keras")
        prediction = model.predict(review_vector)
        prediction = 1 if prediction >= 0.5 else 0
        if prediction == 1:
            print(f"Metin Sınıfı 1: Saldırganlık, zorbalık veya linç tespit edildi.")
        else:
            print("Metin Sınıfı 0: Saldırganlık, zorbalık veya linç tespit edilemedi.")
    except FileNotFoundError:
        print(f"İlgili model dosyası bulunamadı. Lütfen dlModel_LSTM() fonksiyonunu kullandığınızdan ve dosya varlığından emin olun.")