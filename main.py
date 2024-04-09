import numpy as np
import pandas as pd
import Process
import FeatureExtraction
import Model
import os
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

# CONTROL FOR CRUDE, CLEAN, NORMAL, PROCESSED DATASETS and PROCESSING DATASETS
if os.path.exists("datasets/processed/train.csv") and os.path.exists("datasets/processed/test.csv") and os.path.exists(
        "datasets/processed/valid.csv"):
    #print("Crude Dataset already exists")
    pass
else:
    print("No Processed Dataset found, being processed...")
    Process.processDataset()
if os.path.exists("datasets/processed/clean_train.csv") and os.path.exists(
        "datasets/processed/clean_test.csv") and os.path.exists("datasets/processed/clean_valid.csv"):
    #print("Clear Dataset already exists")
    pass
else:
    print("No Cleaned Dataset found, being cleaned...")
    Process.processData()
if os.path.exists("datasets/processed/normal_train.csv") and os.path.exists(
        "datasets/processed/normal_test.csv") and os.path.exists("datasets/processed/normal_valid.csv"):
    #print("Normalized Dataset already exists")
    pass
else:
    print("No Normalized Dataset found, being normalized...")
    Process.normalizeData()
if os.path.exists("datasets/processed/processed_train.csv") and os.path.exists(
        "datasets/processed/processed_test.csv") and os.path.exists("datasets/processed/processed_valid.csv"):
    #print("Processed Dataset already exists. It's ready for train.")
    pass
else:
    print("No Processed Dataset found, Stem finding process being...")
    Process.stemmingData()

dataset_name = "processed"
# READING DATA FROM 'PROCESSED' DATASET
df_train = pd.read_csv(f"datasets/processed/{dataset_name}_train.csv")
df_train = df_train.dropna()
x_train = df_train["text"].values
y_train = df_train["label"].values

df_test = pd.read_csv(f"datasets/processed/{dataset_name}_test.csv")
df_test = df_test.dropna()
x_test = df_test["text"].values
y_test = df_test["label"].values

df_valid = pd.read_csv(f"datasets/processed/{dataset_name}_valid.csv")
df_valid = df_valid.dropna()
x_valid = df_valid["text"].values
y_valid = df_valid["label"].values

"""""control_train = x_train.copy()
control_test = x_test.copy()
control_valid = x_valid.copy()
print(f"First read : {x_train.shape}")
control_train, control_test, control_valid = FeatureExtraction.vectorizeTfidf(control_train,control_test,control_valid) #vectorize the text contents with tfidf vectorizer
print(f"After tfidf: {control_train.shape}")
#print(f"control_train: {control_train}")
#print(f"control_test: {control_test}")
print(type(control_train))
print(control_train[0])

x_train, x_test, x_valid = FeatureExtraction.vectorizeFastText(x_train,x_test,x_valid)
x_train = np.array(x_train)
x_test = np.array(x_test)
x_valid = np.array(x_valid)
print(f"After doc2vec: {x_train.shape}")
#print(f"x_train: {x_train}")
#print(f"x_test: {x_test}")
print(type(x_train))
print(x_train[0])"""""

def welcome():
    print("""
    **** __     __       _    _ __     __        ____   ****
    ***  \ \   / / _    | |  | |\ \   / / _    / ___|    ***
    **    \ \_/ / / \   | |__| | \ \_/ / / \   | |        **
    **     \   / / _ \  |  __  |  \   / / _ \  \___ \     **
    ***     | | / ___ \ | |  | |   | | / ___ \  ___) |   ***
    ****    |_|/_/   \_\|_|  |_|   |_|/_/   \_\|____/   ****
    """)


def main_menu(x_train,x_test,x_valid,y_train,y_test,y_valid):
    print("________________________________")
    welcome()
    #VECTORIZATION : Feature Extraction
    x_train, x_test, x_valid, vectorization_method_choice = vectorization_menu(x_train, x_test, x_valid)

    """""
    #SCALING : Worse 
    sc = StandardScaler(with_mean=False)
    sc.fit_transform(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    x_valid = sc.transform(x_valid)
    """""

    #TRAINING
    model_choice = train_menu(x_train, y_train, x_test, y_test, x_valid, y_valid)

    #PREDICTION
    print("3. Predict with Trained Model")
    review = [input("Review:")]
    review = Process.processUserReview(review)
    if (vectorization_method_choice == "1"):
        while not (review[0] == "" or review[0].isspace()):
            review_vector = FeatureExtraction.vectorizeTfidf_Transform(review)         #Tfidf transform
            prediction_menu(review_vector,model_choice)
            review = [input("Review:")]
            review = Process.processUserReview(review)
    elif (vectorization_method_choice == "2"):
        while not (review[0] == "" or review[0].isspace()):
            review_vector = FeatureExtraction.vectorizeCV_Transform(review)            #Count vectorizer transform
            prediction_menu(review_vector, model_choice)
            review = [input("Review:")]
            review = Process.processUserReview(review)
    elif (vectorization_method_choice == "3"):
        while not (review[0] == "" or review[0].isspace()):
            review_vector = FeatureExtraction.vectorizeDoc2Vec_Transform(review)     #Doc2Vec transform
            # Use this code if you using FastText or Doc2Vec for vectorization
            review_vector = np.array(review_vector)
            review_vector = csr_matrix(review_vector)
            prediction_menu(review_vector, model_choice)
            review = [input("Review:")]
            review = Process.processUserReview(review)
    elif (vectorization_method_choice == "4"):
        while not (review[0] == "" or review[0].isspace()):
            review_vector = FeatureExtraction.vectorizeFastText_Transform(review)      #Fasttext transform
            # Use this code if you using FastText or Doc2Vec for vectorization
            review_vector = np.array(review_vector)
            review_vector = csr_matrix(review_vector)
            prediction_menu(review_vector, model_choice)
            review = [input("Review:")]
            review = Process.processUserReview(review)
    elif (vectorization_method_choice == "5"):
        while not (review[0] == "" or review[0].isspace()):
            review_vector = FeatureExtraction.vectorize_FastText_Pretrained_Transform(review)  #Fasttext pretrained model transform
            # Use this code if you using FastText or Doc2Vec for vectorization
            review_vector = np.array(review_vector)
            review_vector = csr_matrix(review_vector)
            prediction_menu(review_vector, model_choice)
            review = [input("Review:")]
            review = Process.processUserReview(review)
    else:
        print("Couldn't read the vectorization method!")





def change_shape(x_train, x_test, x_valid):
    # Use this code if you using FastText or Doc2Vec for vectorization
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_valid = np.array(x_valid)

    # For convert np.arrays to sparse matrix. This code can used for convert vectors from fasttext and Doc2Vec to tfidf format(sparse matrix). You can try for increase model accuracy
    x_train = csr_matrix(x_train)
    x_test = csr_matrix(x_test)
    x_valid = csr_matrix(x_valid)

    return x_train, x_test, x_valid

def vectorization_menu(x_train, x_test, x_valid):
    print("------------------------------")
    print("Select a vectorization method:")
    print("1. TF-IDF Vectorizer")
    print("2. Count Vectorizer")
    print("3. Doc2Vec")
    print("4. FastText")
    print("5. Pretrained FastText")
    choice = input("Enter your choice: ")

    if choice == "1":
        # Vectorize the text contents with TF-IDF Vectorizer
        x_train, x_test, x_valid = FeatureExtraction.vectorizeTfidf(x_train, x_test, x_valid)
    elif choice == "2":
        # Vectorize the text contents with Count Vectorizer
        x_train, x_test, x_valid = FeatureExtraction.vectorizeCV(x_train, x_test, x_valid)
    elif choice == "3":
        # Vectorize the text contents with Doc2Vec
        x_train, x_test, x_valid = FeatureExtraction.vectorizeDoc2Vec(x_train, x_test, x_valid)
        x_train, x_test, x_valid = change_shape(x_train, x_test, x_valid)
    elif choice == "4":
        # Vectorize the text contents with FastText from gensim
        x_train, x_test, x_valid = FeatureExtraction.vectorizeFastText(x_train, x_test, x_valid)
        x_train, x_test, x_valid = change_shape(x_train, x_test, x_valid)
    elif choice == "5":
        # Vectorize the text contents with Pretrained FastText
        if os.path.exists("D:/NLP_Models/cc.tr.300.bin"):
            x_train, x_test, x_valid = FeatureExtraction.vectorize_FastText_Pretrained(x_train, x_test, x_valid)
            x_train, x_test, x_valid = change_shape(x_train, x_test, x_valid)
        else:
            print("\033[91mPretrained fasttext model doesn't exist. Please choose another option!\033[0m")
            x_train, x_test,x_valid, choice = vectorization_menu(x_train,x_test,x_valid)
    else:
        print("Invalid choice. Please select a valid option.")
        x_train, x_test,x_valid, choice = vectorization_menu(x_train,x_test,x_valid)

    return x_train, x_test, x_valid, choice


def train_menu(x_train, y_train, x_test, y_test, x_valid, y_valid):
    print("------------------------------")
    print("Select a model for training:")
    print("1. Logistic Regression")
    print("2. Naive Bayes")
    print("3. Random Forest")
    print("4. Support Vector Machine")
    print("5. Decision Tree")
    print("6. Deep Learning")
    #print("7. Deep Learning 2 (For FastText and Doc2Vec vectors)")
    #print("8. Deep Learning with LSTM (DL-LSTM)")
    choice = input("Enter your choice: ")

    if choice == "1":
        # Train a logistic regression model
        Model.lrModel(x_train, y_train, x_test, y_test)
    elif choice == "2":
        # Train a Naive Bayes model
        Model.nbModel(x_train, y_train, x_test, y_test)
    elif choice == "3":
        # Train a random forest model
        Model.rfModel(x_train, y_train, x_test, y_test)
    elif choice == "4":
        # Train a support vector machine model
        Model.svmModel(x_train, y_train, x_test, y_test)
    elif choice == "5":
        # Train a decision tree model
        Model.dtModel(x_train, y_train, x_test, y_test)
    elif choice == "6":
        # Train a deep learning model
        Model.dlModel(x_train, y_train, x_test, y_test, x_valid, y_valid)
    else:
        print("Invalid choice. Please select a valid option.")
        choice = train_menu(x_train,y_train,x_test,y_test,x_valid,y_valid)
    """""elif choice == "7":
            Model.dlModel_2(x_train, y_train, x_test, y_test, x_valid, y_valid)
        elif choice == "8":
            # Train a deep learning model with LSTM
            Model.dlModel_LSTM(x_train, y_train, x_test, y_test, x_valid, y_valid)"""""
    return choice


def prediction_menu(review_vector,model_choice):
    if(model_choice == "1"):
        Model.lrPredict(review_vector)
    elif(model_choice == "2"):
        Model.nbPredict(review_vector)
    elif (model_choice == "3"):
        Model.rfPredict(review_vector)
    elif (model_choice == "4"):
        Model.svmPredict(review_vector)
    elif (model_choice == "5"):
        Model.dtPredict(review_vector)
    elif (model_choice == "6"):
        Model.dlPredict(review_vector)
    """""elif (model_choice == "7"):
        Model.dlPredict_2(review_vector)
    elif (model_choice == "8"):
        Model.dlPredict_LSTM(review_vector)"""""




if __name__ == "__main__":
    while True:
        main_menu(x_train,x_test,x_valid,y_train,y_test,y_valid)