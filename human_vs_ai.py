import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Models:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.punctuations = string.punctuation + "0123456789"
        self.clf = None
        # self.train = pd.read_csv(r"data/Poem_classification_train_data.csv")
        # self.train.dropna(inplace=True)
        # self.test = pd.read_csv(r"data/Poem_classification_test_data.csv")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pred = None
        self.model = model

        
    def model_init(self):
        if self.model == "random forest":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), RandomForestClassifier(random_state=42))
        elif self.model == "svm":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), SVC(random_state=42))
        elif self.model == "xgboost":
            # self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), XGBClassifier())
            self.clf = make_pipeline(CountVectorizer(stop_words="english"), XGBClassifier())
        elif self.model == "decision tree":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), DecisionTreeClassifier(random_state=42))
        elif self.model == "naive bayes multinomial":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), MultinomialNB())
        elif self.model == "naive bayes complement":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), ComplementNB())
        elif self.model == "knn":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), KNeighborsClassifier())
        if self.model != "xgboost":
            self.clf.fit(self.X_train, self.y_train)
        else:
            self.clf.fit(self.X_train, self.xg_boost()[0])
        # print(f"{type(self.clf) = }")

    def xg_boost(self):
        le = LabelEncoder()
        y_train_lr = le.fit_transform(self.y_train)
        y_test_lr = le.transform(self.y_test)
        classes = le.classes_
        # print(f"{classes = }")
        return y_train_lr, y_test_lr, classes

    def metrics(self):
        self.pred = self.clf.predict(self.X_test)
        classes = self.clf.classes_
        if self.model == "xgboost":
            class_report = pd.DataFrame(classification_report(self.xg_boost()[1], self.pred, target_names=self.xg_boost()[2], output_dict=True)).transpose()
        else:
            class_report = pd.DataFrame(classification_report(self.y_test, self.pred, target_names=classes, output_dict=True)).transpose()
        return class_report

    def remove_punctuations(self, series):
        res = []
        for _ in series:
            res.append("".join([char for char in _ if char not in self.punctuations]))
        return res

    def save_fig(self):
        if self.model == "xgboost":
            cm = confusion_matrix(self.xg_boost()[1], self.pred)
            display = ConfusionMatrixDisplay(cm, display_labels=self.xg_boost()[2]).plot()
        else:
            cm = confusion_matrix(self.y_test, self.pred)
            display = ConfusionMatrixDisplay(cm, display_labels=self.clf.classes_).plot()
        plt.savefig("temp.png")

    def final_result(self):
        self.model_init()
        self.metrics()
        self.save_fig()