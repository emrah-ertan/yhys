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