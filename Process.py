import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from trnlp import TrnlpWord

kumeler = ["train","test","valid"]
kufurler = pd.read_csv("datasets/kufurler.csv")
kufurler = kufurler["kufurler"].tolist()
noktalama = string.punctuation
etkisiz = set(stopwords.words("turkish"))
obj = TrnlpWord()

#Ham veri kümesindeki id sütununu siler
def processDataset():
    for sira in kumeler:
        try:
            df  = pd.read_csv(f"datasets/crude/{sira}.csv")
            df = df.drop("id",axis=1)
            df.to_csv(f"datasets/processed/{sira}.csv",index=False)
        except FileNotFoundError:
            print(f"processDataset() fonksiyonunda bir hata meydana geldi !\n{FileNotFoundError}")


#Veri kümesindeki verilerin ön işleme sürecidir.
def processData():
    for sira in kumeler:
        try:
            df = pd.read_csv(f"datasets/processed/{sira}.csv")
            cleaned_reviews = []
            for review in df['text']:
                review = re.sub(r"'", "", review)
                review = review.replace("I", "ı")
                #kullanıcı etiketlemeler kaldırılacak : @ertan
                if "@" in review:
                    kelimeler = review.split()
                    kelimeler = [kelime for kelime in kelimeler if not kelime.startswith("@")]
                    review = " ".join(kelimeler)
                review = re.sub('[^a-zA-ZığüşöçİĞÜŞÖÇ0123456789 ]', '', review)
                review = review.lower()
                # Stopwords temizliği
                cleaned_words = [word for word in review.split() if word.lower() not in etkisiz and not word.isnumeric() and word != "user"]
                cleaned_review = ' '.join(cleaned_words)
                # Noktalama işareti temizliği
                cleaned_review = ''.join([char for char in cleaned_review if char not in noktalama])
                cleaned_reviews.append(cleaned_review)
            #cleaned_reviews listesi üzerinde normalizasyon, stemming işlemleri yapılacak (sırasıyla yapoılarak denencek)
            df['text'] = cleaned_reviews
            df.to_csv(f'datasets/processed/clean_{sira}.csv', index=False)
        except FileNotFoundError:
            print(f"processData() fonksiyonunda bir hata meydana geldi ! \n{FileNotFoundError}")


#Veri kümesindeki anormallikler incelendi. Normalizasyon kütüphaneleri veriyi bozduğundan yapılan yazım yanlışları manuel düzeltiliyor.
def normalizeData():
    for sira in kumeler:
        df = pd.read_csv(f"datasets/processed/clean_{sira}.csv")
        df = df.copy()
        cumleler = df["text"].tolist()
        for i in range(len(cumleler)):
            cumle = list(cumleler[i])
            for ch in range(len(cumle)):
                if ch == 0:
                    if cumle[ch + 1] == " ":
                        cumle[ch + 1] = ""
                        cumle[ch] = ""
                    elif cumle[ch + 1] == cumle[ch]:
                        cumle[ch + 1] = ""
                elif ch == len(cumle) - 1:
                    if cumle[ch - 1] == " ":
                        cumle[ch - 1] = ""
                        cumle[ch] = ""
                    elif cumle[ch - 1] == cumle[ch]:
                        cumle[ch] = ""
                elif ch != len(cumle) - 1 and ch != 0:
                    if cumle[ch - 1] == " " and cumle[ch + 1] == " ":
                        cumle[ch - 1] = ""
                        cumle[ch + 1] = ""
                        cumle[ch] = " "
                    elif cumle[ch - 1] == cumle[ch] and cumle[ch + 1] == cumle[ch]:
                        cumle[ch] = ""
            cumleler[i] = "".join(cumle)
        df["text"] = cumleler
        df.to_csv(f"datasets/processed/normal_{sira}.csv",index = False)





#veri kümesindeki kelimelerin kök değerlerini alarak köklerden oluşan yeni veri kümesi oluşturur
def stemmingData():
    for sira in kumeler:
        df = pd.read_csv(f"datasets/processed/normal_{sira}.csv")
        processedReviews = []
        for review in df["text"]:
            # normalizasyon burada olacak
            # Kelimelerin ayrılması ve köklerin bulunması
            kelimelerList = review.split()
            stemList = []
            for kelime in kelimelerList:
                # Kelimenin kökünü alıyoruz
                if kelime not in kufurler:
                    obj.setword(kelime)
                    stem = obj.get_stem  # İlk gözlemlerde get_stem, get_base'e göre daha iyi sonuç veriyor. Detaylı incelemedim
                    # stem = obj.get_base
                    stem = stem.replace("â", "a").replace("ê", "e").replace("î", "ı").replace("ô", "o").replace("û", "u")
                    #print(f"get_stem:{stem}")                  #get_stem, get_base'den daha iyi sonuç verdiğinden get_stem kullanılıyor
                    # print(f"get_base:{obj.get_base}")
                    if stem.isspace():
                        pass
                    else:
                        stemList.append(stem)
                else:
                    stemList.append(kelime)
            # İşlenmiş cümleyi listeye ekliyoruz
            yorum = " ".join(stemList).strip()
            yorum = yorum.replace("I", "ı").lower()
            yorum = " ".join(yorum.split())
            processedReviews.append(yorum)
        # print(len(processedReviews))
        # print(len(stemList))
        df["text"] = processedReviews
        df.to_csv(f"datasets/processed/processed_{sira}.csv", index=False)







def processUserReview(userReview):
    #userInput ilk işleme
    cleaned_reviews = []
    for review in userReview:
        review = re.sub(r"'", "", review)
        review = review.replace("I", "ı")
        # kullanıcı etiketlemeler kaldırılacak : @ertan
        if "@" in review:
            kelimeler = review.split()
            kelimeler = [kelime for kelime in kelimeler if not kelime.startswith("@")]
            review = " ".join(kelimeler)
        review = re.sub('[^a-zA-ZığüşöçİĞÜŞÖÇ0123456789]', ' ', review)
        review = review.lower()
        # Stopwords temizliği
        cleaned_words = [word for word in review.split() if
                         word.lower() not in etkisiz and not word.isnumeric() and word != "user"]
        cleaned_review = ' '.join(cleaned_words)
        # Noktalama işareti temizliği
        cleaned_review = ''.join([char for char in cleaned_review if char not in noktalama])
        cleaned_reviews.append(cleaned_review)


        #userInput normalizasyonu
        cumleler = cleaned_reviews
        for i in range(len(cumleler)):
            cumle = list(cumleler[i])
            for ch in range(len(cumle)):
                if ch == 0:
                    try:
                        if cumle[ch + 1] == " ":
                            cumle[ch + 1] = ""
                            cumle[ch] = ""
                        elif cumle[ch + 1] == cumle[ch]:
                            cumle[ch + 1] = ""
                    except:
                        pass
                elif ch == len(cumle) - 1:
                    try:
                        if cumle[ch - 1] == " ":
                            cumle[ch - 1] = ""
                            cumle[ch] = ""
                        elif cumle[ch - 1] == cumle[ch]:
                            cumle[ch] = ""
                    except:
                        pass
                elif ch != len(cumle) - 1 and ch != 0:
                    try:
                        if cumle[ch - 1] == " " and cumle[ch + 1] == " ":
                            cumle[ch - 1] = ""
                            cumle[ch + 1] = ""
                            cumle[ch] = " "
                        elif cumle[ch - 1] == cumle[ch] and cumle[ch + 1] == cumle[ch]:
                            cumle[ch] = ""
                    except:
                        pass
            cleaned_reviews = "".join(cumle)

        #userInput stemming
        processedReviews = []
        review = cleaned_reviews
        kelimelerList = review.split()
        stemList = []
        for kelime in kelimelerList:
            # Kelimenin kökünü alıyoruz
            if kelime not in kufurler:
                obj.setword(kelime)
                stem = obj.get_stem  # İlk gözlemlerde get_stem, get_base'e göre daha iyi sonuç veriyor. Detaylı incelemedim
                # stem = obj.get_base
                stem = stem.replace("â", "a").replace("ê", "e").replace("î", "ı").replace("ô", "o").replace("û",                                                                                                          "u")
                # print(f"get_stem:{stem}")                  #get_stem, get_base'den daha iyi sonuç verdiğinden get_stem kullanılıyor
                # print(f"get_base:{obj.get_base}")
                if stem.isspace():
                    pass
                else:
                    stemList.append(stem)
            else:
                stemList.append(kelime)
        # İşlenmiş cümleyi listeye ekliyoruz
        yorum = " ".join(stemList).strip()
        yorum = yorum.replace("I", "ı").lower()
        yorum = " ".join(yorum.split())
        processedReviews.append(yorum)
        cleaned_reviews = processedReviews
    return cleaned_reviews



def tokenize(data_list):    # Input type is list, output type is
    tokenized_phrases = []
    for phrase in data_list:
        tokens = [word_tokenize(word) for word in phrase.split()]
        tokenized_phrases.append(tokens)
    return word_tokenize(data_list)