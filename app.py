import streamlit as st
from human_vs_ai import Models
import streamlit.components.v1 as components

st.markdown("<h1 style='text-align: center; color: white;'>Classifying Poems</h1>", unsafe_allow_html=True)

text = st.text_input("Enter the text to classify.")
# choice = st.selectbox("Pick your preferred model.", ["random forest", "svm", "xgboost", "decision tree", "naive bayes multinomial", "naive bayes complement", "knn", "logreg"])
mapper = {"Logistic Regression": "logreg"}
choice = st.selectbox("Pick your preferred model.", ["Logistic Regression"])

model = Models("./AI_Human.csv", f"{mapper[choice]}")
model.final_result(False)

predict_button = st.button("Classify Text")

if predict_button:
    if choice == "xgboost":
        classes = model.xg_boost()[2]
        result = classes[model.clf.predict([text])]
    else:
        result = model.clf.predict([text])
    st.text(result[0])
    
    st.image(f"{mapper[choice]}png")

    
    