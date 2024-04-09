import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns



def showGraphic(vectors, title="VECTORS GRAPHIC"):
    svd = TruncatedSVD(n_components=2)
    x_svd = svd.fit_transform(vectors)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_svd[:, 0], x_svd[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def showClasses(x_vectors, labels, title="CLASSES GRAPHIC"):
    svd = TruncatedSVD(n_components=2)
    x_svd = svd.fit_transform(x_vectors)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_svd[:, 0], x_svd[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()


def showConfusionMatrix(cm, classes=['NOT CYBERBULLYING', 'CYBERBULLYING']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def showAccuracy(accuracies_list,vectorization_method):
    models = ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVM', 'Decision Tree', 'Deep Learning']
    colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'lightcoral', 'lightblue']  # Specify colors for each model

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies_list, color=colors)  # Use the colors list for column colors
    plt.xlabel('AI Models')
    plt.ylabel('Accuracy Scores')
    plt.title(f'Accuracy Scores of AI Models for {vectorization_method}')
    plt.ylim(0.75, 1.0)  # Set y-axis limit to start from 0.8
    plt.show()

accuracies_list_tfidf =[0.9016504634863215,0.8852588740673751,0.9001808727108298,0.9032330997060819,0.8703368754239205,0.9032331109046936]
accuracies_list_CV = [0.9023287361519331,0.8902328736151933,0.8974677820483834,0.8985982364910694,0.8640063305448791,0.9035722613334656]

#showAccuracy(accuracies_list_tfidf,"TFIDF")
#showAccuracy(accuracies_list_CV,"Count Vectorizer")