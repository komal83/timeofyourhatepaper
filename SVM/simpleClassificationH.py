import pandas as pd
import os
import sys
import string
import spacy
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from spacy.lang.it.stop_words import STOP_WORDS
from spacy.lang.it import Italian
from sklearn import svm
from sklearn import neural_network
from sklearn import metrics

punctuations = string.punctuation
nlp = spacy.load("it_core_news_sm")
stop_words = spacy.lang.it.stop_words.STOP_WORDS
parser = Italian()

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

# Tokenizer function
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.text for word in mytokens ]
    # remove stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    # return preprocessed list of tokens
    return mytokens

def printNMostInformative(vectorizer, clf, N):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)

def main():
    filelist = os.listdir(sys.argv[1])
    filelist.sort()
    fi = 0
    resultList = []
    df_haspeede = pd.read_csv("training_new.csv", sep=",", header=None, names=["id","text","label"])
    while fi < len(filelist)-1:
        print("TRAIN: "+filelist[fi])
        print("========================================");
        df_train=pd.read_csv(sys.argv[1] + "/" + filelist[fi], sep=",", header=None, names=["id","text","label"])
        frames = [df_haspeede, df_train]
        df_train = pd.concat(frames)
        print("Build model...")
        bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
        #tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
        X_train = df_train['text'] # the features
        y_train = df_train['label'] # the labels
        classifier = svm.LinearSVC(max_iter=10000)
        #classifier = neural_network.MLPClassifier()
        pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 #('vectorizer', tfidf_vector),
                 ('classifier', classifier)])
        pipe.fit(X_train,y_train)
        print("TEST: "+filelist[fi + 1])
        print("========================================");
        df_test=pd.read_csv(sys.argv[1] + "/" + filelist[fi + 1], sep=",", header=None, names=["id","text","label"])
        #df_test.shape
        #df_test.info()
        print("Evaluate...")
        X_test = df_test['text'] # the features
        y_test = df_test['label'] # the labels
        predicted = pipe.predict(X_test)
        acc = metrics.accuracy_score(y_test, predicted)
        print("Accuracy:", acc)
        precision = metrics.precision_score(y_test, predicted)
        #save prediction
        fpred = open('prediction_'+filelist[fi], 'w')
        for p in predicted:
			fpred.write("%d"%p)
			fpred.write('\n')
        fpred.close()
        #print("Precision:", precision)
        recall = metrics.recall_score(y_test, predicted)
        #print("Recall:", recall)
        fm = metrics.f1_score(y_test, predicted)
        #print("F-measure:", fm)
        #resultList.append((acc, precision, recall, fm))
        resultList.append(metrics.classification_report(y_test, predicted))
        #print(metrics.classification_report(y_test, predicted))
        printNMostInformative(bow_vector, classifier, 100)
        print("========================================")
        fi = fi + 1
    print("Results:")
    for res in resultList:
        print(res)

if __name__ == '__main__':
    main()
