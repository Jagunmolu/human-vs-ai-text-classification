import joblib

def predict(data, model):
    clf = joblib.load(f"{model}.sav")
    return clf.predict(data)