import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os

@st.cache_resource
def load_and_train():
    # show files available in Streamlit Cloud
    st.write("Available files:", os.listdir())

    # ✅ RELATIVE PATH ONLY
    df = pd.read_csv("movie_interests_decisiontree.csv")

    le_genre = LabelEncoder()
    df["Genre_encoded"] = le_genre.fit_transform(df["Genre"])

    X = df[["Age", "Genre_encoded"]]
    y = df["Interest"]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    return model, le_genre, df


st.title("🎵 Music Subscription Plan Predictor")

model, le_genre, df = load_and_train()

age = st.slider("Age", 10, 70, 25)
genre = st.selectbox("Preferred Genre", df["Genre"].unique())

genre_encoded = le_genre.transform([genre])[0]
prediction = model.predict([[age, genre_encoded]])

st.success(f"Recommended Plan: {prediction[0]}")
