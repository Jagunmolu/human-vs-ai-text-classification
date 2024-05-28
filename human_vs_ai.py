import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
# from sklearn.preprocessing import FunctionTransformer
sns.set()

import pandas as pd
import matplotlib.pyplot as plt
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


class Models:
    def __init__(self, file_path, model):
        self.punctuations = string.punctuation + "0123456789"
        self.clf = None
        self.file_path = file_path
        data = self.split_data()
        self.X_train = data[0]
        self.y_train = data[2]
        self.X_test = data[1]
        self.y_test = data[3]
        # print(self.X_train.shape, self.y_train.shape)
        self.pred = None
        self.model = model

    def split_data(self):
        data = self.wrangle()
        # X = data.drop("generated", axis=1)
        X = data["text"]
        y = data["generated"]
        # print(X.shape, y.shape)
        return train_test_split(X, y, test_size=.25)

    def selected_index(self, arr, size=150_000):
        np.random.seed(42)
        return np.random.choice(arr, size=size, replace=False)
        
    def wrangle(self):
        data = pd.read_csv(self.file_path)
        data["generated"] = data["generated"].astype(int)
        data = data.dropna()
        length = data.shape[0]
        np.random.seed(42)
        gen0 = data.iloc[self.selected_index(data[data["generated"] == 0].index, 20_000)]
        gen1 = data.iloc[self.selected_index(data[data["generated"] == 1].index, 20_000)]
        data1 = pd.concat([gen0, gen1], axis=0).reset_index(drop=True)
        idx = np.array(data1.index)
        np.random.shuffle(idx)
        data1 = data1.iloc[idx].reset_index(drop=True)
        # data1.to_csv("./ai_vs_human.csv", index=False)
        return data1

    def random_forest(self):
        pass

    def svm(self):
        pass

    def xgboost(self):
        pass

    def dtree(self):
        pass

    def nbmul(self):
        pass

    def nbcom(self):
        pass

    def knn(self):
        pass

    def logreg(self):
        pass

        
    def model_init(self, from_file=False):
        if from_file:
            self.clf = joblib.load(f"{self.model}.sav")
            self.clf.fit(self.X_train, self.y_train)
        else:
            if self.model == "random forest":
                self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), RandomForestClassifier(random_state=42))
            elif self.model == "svm":
                self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), SVC(random_state=42))
            elif self.model == "xgboost":
                self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), XGBClassifier())
            elif self.model == "decision tree":
                self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), DecisionTreeClassifier(random_state=42))
            elif self.model == "naive bayes multinomial":
                self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), MultinomialNB())
            elif self.model == "naive bayes complement":
                self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), ComplementNB())
            elif self.model == "knn":
                self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), KNeighborsClassifier())
            elif self.model == "logreg":
                self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), LogisticRegression(max_iter=100_000))
        
            self.clf.fit(self.X_train, self.y_train)
            joblib.dump(self.clf, f"{self.model}.sav")

    def xg_boost(self):
        le = LabelEncoder()
        y_train_lr = le.fit_transform(self.y_train)
        y_test_lr = le.transform(self.y_test)
        classes = le.classes_
        return y_train_lr, y_test_lr, classes

    def metrics(self):
        self.pred = self.clf.predict(self.X_test)
        # classes = self.clf.classes_
        classes = ["Human", "AI"]
        # if self.model == "xgboost":
        #     class_report = pd.DataFrame(classification_report(self.xg_boost()[1], self.pred, target_names=self.xg_boost()[2], output_dict=True)).transpose()
        # else:
        class_report = pd.DataFrame(classification_report(self.y_test, self.pred, target_names=classes, output_dict=True)).transpose()
        return class_report

    def remove_punctuations(self, series):
        res = []
        for _ in series:
            res.append("".join([char for char in _ if char not in self.punctuations]))
        return res

    def save_fig(self):
        # plt.figure(figsize=(10,5))
        # if self.model == "xgboost":
        #     cm = confusion_matrix(self.xg_boost()[1], self.pred)
        #     display = ConfusionMatrixDisplay(cm, display_labels=self.xg_boost()[2]).plot()
        # else:
        cm = confusion_matrix(self.y_test, self.pred)
        display = ConfusionMatrixDisplay(cm, display_labels=self.clf.classes_)
        display.plot()
        plt.grid(False)
        plt.savefig(f"{self.model}.png")
        plt.close()

    def final_result(self, from_file):
        print(f"Computing for {self.model}")
        self.model_init(from_file=from_file)
        self.metrics()
        self.save_fig()