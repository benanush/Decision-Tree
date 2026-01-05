import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os

@st.cache_resource
def load_and_train():
    # DEBUG: show files in cloud
    st.write("Files in directory:", os.listdir())

    # ✅ CORRECT PATH (relative)
    df = pd.read_csv("Movie_Interests_DecisionTree.csv")

    le_genre = LabelEncoder()
    df["Genre_encoded"] = le_genre.fit_transform(df["Genre"])

    X = df[["Age", "Genre_encoded"]]
    y = df["Interest"]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    return model, le_genre, df


st.title("🎬 Movie Interest Prediction")

model, le_genre, df = load_and_train()

age = st.slider("Select Age", 10, 70, 25)
genre = st.selectbox("Select Genre", df["Genre"].unique())

genre_encoded = le_genre.transform([genre])[0]
prediction = model.predict([[age, genre_encoded]])

st.success(f"Predicted Interest: {prediction[0]}")
