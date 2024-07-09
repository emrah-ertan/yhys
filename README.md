# YaHYaS (Ya hayır konuş ya sus)

## İçindekiler
- [Giriş](#giriş)
- [Yöntem](#yöntem)
- [Bulgular](#bulgular)
- [Sonuç](#sonuç)

## GİRİŞ

Günümüzde sosyal medya uygulamalarının yaygın şekilde kullanımı, insanların birbirleri ile olan etkileşimini fazlasıyla artırmış durumdadır. Bu durum iyiye kullanıldığında oldukça güzel gelişmelere sebebiyet verebilirken, maalesef bu durum kötüye kullanıldığında insanlar üzerinde zannedilenden çok daha olumsuz sonuçlar doğurabilmektedir. İnsanların psikolojik sağlıklarını olumsuz etkileyebilecek şekilde yapılan yorumlar toplum yapısına da zarar vermektedir. Bu durumların önüne geçilebilmesi amacıyla projemde bu sorunu çözmek adına makine öğrenmesi algoritmaları ve derin sinir ağları modeli kullanarak metinlerde siber zorbalık tespiti yapmayı amaçladım.

Proje için kullanılan teknolojiler aşağıdaki gibidir:

- Python
- PyCharm
- Pandas
- Nltk
- Trnlp
- Gensim
- Sklearn
- Keras
- Numpy
- Matplotlib

## YÖNTEM

İlk olarak verilerin ham haliyle bulunduğu veri kümelerinde ilgili temizliğin ve gerekli ön işleme adımlarının yapılması için kullanılan fonksiyonlar `Process.py` dosyası içerisinde yer almaktadır. Ön işleme için ham veri kümesinde kullanılmayan id sütununun kaldırılması, Türkçe karakterlerin korunarak istenmeyen karakterlerin metin verilerinden kaldırılması, küçük harf dönüşümü, yanlış yazımı düzeltmek amacıyla normalizasyon işlemleri ve son olarak kök bulma (stemming) işlemleri uygulanmıştır.

Öznitelik Çıkarımı için işlenmiş veri kümesinden alınan verilerin vektörleştirme işlemi için kullanılan fonksiyonlar `FeatureExtraction.py` dosyası içerisinde yer almaktadır. Vektörizasyon için aşağıdaki metotlar ayrı ayrı uygulanmıştır:

- Term Frequency Inverse Document Frequency
- Count Vectorizer
- Doc2Vec
- Fasttext

Eğitim aşaması için oluşturulan vektör değerlerinin ilgili modellerin oluşturulması, modellerde eğitim, test ve doğrulama için gerekli fonksiyonlar `Model.py` dosyası içerisinde yer almaktadır. Bu aşamada aşağıdaki metotlar ayrı ayrı uygulanmıştır:

- Logistic Regression
- Naive Bayes
- Random Forest
- Support Vector Machine
- Decision Tree
- Deep Learning

## BULGULAR

Yapılan testler sonucunda, vektörizasyon yöntemlerinin yapay zeka algoritmaları üzerineki performansları hesaplanmıştır. 

![Tfidf Accuracy](img/Figure_1.png)
![Count Vectorizer Accuracy](img/Figure_2.png)

Bu verilere ek olarak, FastText ve Doc2Vec yöntemleri ile eğitilen derin sinir ağı modeli yaklaşık olarak 0.78 ve 0.82 doğruluk değerlerinde kalmışlardır. Bu sebepten dolayı 'Tfidf' ve 'Count Vectorizer' vektörleştirme yöntemlerinin kullanımı daha doğrudur. Ayrıca eğitim sonucunda ortaya çıkan 'confusion matrix' değerleri aşağıdaki gibidir (LSTM için vektörleştirme Embedding katmanı ile gerçekleştirilmiştir):

**Logistic Regression**
<p float="left">
  <img src="img/tfidf_lr.png" width="450" />
  <img src="img/cv_lr.png" width="450" />
</p>

**Naive Bayes**
<p float="left">
  <img src="img/tfidf_nb.png" width="450" />
  <img src="img/cv_nb.png" width="450" />
</p>

**Random Forest**
<p float="left">
  <img src="img/tfidf_rf.png" width="450" />
  <img src="img/cv_rf.png" width="450" />
</p>

**Support Vector Machine**
<p float="left">
  <img src="img/tfidf_svm.png" width="450" />
  <img src="img/cv_svm.png" width="450" />
</p>

**Decision Tree**
<p float="left">
  <img src="img/tfidf_dt.png" width="450" />
  <img src="img/cv_dt.png" width="450" />
</p>

**Deep Learning**
<p float="left">
  <img src="img/tfidf_dl.png" width="450" />
  <img src="img/cv_dl.png" width="450" />
</p>

**LSTM**
<p float="left">
  <img src="img/cm_lstm.png width="450" />
</p>

## SONUÇ

Siber zorbalığın ve saldırgan yorumların sosyal medya gibi ortamlarda çok fazla görüldüğü bu günlerde, bu gibi istenmeyen durumların tespit edilebilmesini kolaylaştırmak amacıyla çeşitli yöntemler karşılaştırılmıştır. Oluşturulan derin öğrenme modeli %90 üzerinde doğruluk değeri göstermektedir.






# English

## Table of Contents
- [Introduction](#introduction)
- [Method](#method)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

The widespread use of social media applications today has significantly increased people's interaction with each other. When used positively, this situation can lead to very positive developments, but unfortunately, when misused, it can have much more negative effects on people than is thought. Comments that can negatively affect people's psychological health also harm the societal structure. In order to prevent these situations, I aimed to detect cyberbullying in texts using machine learning algorithms and deep neural networks in my project.

The technologies used for the project are as follows:

- Python
- PyCharm
- Pandas
- NLTK
- Trnlp
- Gensim
- Sklearn
- Keras
- Numpy
- Matplotlib

## Method

Firstly, the functions used for the cleaning and preprocessing of the data sets where the raw data is located are in the `Process.py` file. For preprocessing, the following operations were applied to the text data: removing the id column that is not used in the raw data set, removing unwanted characters from the text data while preserving Turkish characters, converting to lowercase, normalization operations to correct misspelled words, and finally stemming operations.

The functions used for vectorizing the data taken from the processed data set for feature extraction are in the `FeatureExtraction.py` file. The following methods were separately applied for vectorization:

- Term Frequency Inverse Document Frequency
- Count Vectorizer
- Doc2Vec
- Fasttext

The functions required for training, testing, and validation in the models created for the training phase are in the `Model.py` file. The following methods were separately applied in this stage:

- Logistic Regression
- Naive Bayes
- Random Forest
- Support Vector Machine
- Decision Tree
- Deep Learning

## Results

As a result of the tests conducted, the performances of the vectorization methods on artificial intelligence algorithms were calculated.

![Tfidf Accuracy](img/Figure_1.png)
![Count Vectorizer Accuracy](img/Figure_2.png)

In addition to these results, the deep neural network model trained with the FastText and Doc2Vec methods remained at approximately 0.78 and 0.82 accuracy values, respectively. Therefore, the use of 'Tfidf' and 'Count Vectorizer' vectorization methods is more accurate. Additionally, the 'confusion matrix' values obtained after the training are as follows (Vectorization for LSTM was implemented with the Embedding layer):

**Logistic Regression**
<p float="left">
  <img src="img/tfidf_lr.png" width="450" />
  <img src="img/cv_lr.png" width="450" />
</p>

**Naive Bayes**
<p float="left">
  <img src="img/tfidf_nb.png" width="450" />
  <img src="img/cv_nb.png" width="450" />
</p>

**Random Forest**
<p float="left">
  <img src="img/tfidf_rf.png" width="450" />
  <img src="img/cv_rf.png" width="450" />
</p>

**Support Vector Machine**
<p float="left">
  <img src="img/tfidf_svm.png" width="450" />
  <img src="img/cv_svm.png" width="450" />
</p>

**Decision Tree**
<p float="left">
  <img src="img/tfidf_dt.png" width="450" />
  <img src="img/cv_dt.png" width="450" />
</p>

**Deep Learning**
<p float="left">
  <img src="img/tfidf_dl.png" width="450" />
  <img src="img/cv_dl.png" width="450" />
</p>

**LSTM**
<p float="left">
  <img src="img/cm_lstm.png width="450" />
</p>

## Conclusion

In these days when cyberbullying and aggressive comments are very common in environments such as social media, various methods have been compared in order to facilitate the detection of such unwanted situations. The deep learning model created achieved an accuracy value of over 90%.
