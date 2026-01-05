import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- Page Config ---
st.set_page_config(page_title="Music Subscription Predictor", layout="wide")

st.title("🎵 Music Subscription Plan Predictor")
st.write("Upload a dataset to predict the best **Subscription Plan** based on user details.")

# --- 1. File Upload ---
uploaded_file = st.file_uploader(
    "📂 Upload Movie_Interests_DecisionTree.csv",
    type=["csv"]
)

# --- 2. Load and Train Model ---
@st.cache_resource
def load_and_train(df):
    le_genre = LabelEncoder()
    df['Genre_Code'] = le_genre.fit_transform(df['Preferred_Genre'])

    X = df[['Age', 'Gender', 'Usage_Hours', 'Genre_Code']]
    y = df['Subscription_Plan']

    model = DecisionTreeClassifier()
    model.fit(X.values, y)

    return model, le_genre, df

# Run only if file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    model, le_genre, df = load_and_train(df)

    # --- 3. Sidebar Inputs ---
    st.sidebar.header("👤 User Information")

    age = st.sidebar.slider("Age", 10, 80, 25)
    gender_text = st.sidebar.selectbox("Gender", ["Male", "Female"])
    usage = st.sidebar.number_input("Daily Usage Hours", 0, 24, 5)
    genre_text = st.sidebar.selectbox("Preferred Genre", le_genre.classes_)

    # --- 4. Encode Inputs ---
    gender_encoded = 1 if gender_text == "Male" else 0
    genre_encoded = le_genre.transform([genre_text])[0]

    # --- 5. Prediction ---
    if st.sidebar.button("Predict Plan"):
        prediction = model.predict([[age, gender_encoded, usage, genre_encoded]])

        st.subheader(f"🎯 Recommended Plan: **{prediction[0]}**")

        if prediction[0] == "Premium":
            st.success("💎 Premium recommended: Best for heavy users!")
        elif prediction[0] == "Basic":
            st.info("⭐ Basic recommended: Balanced and affordable.")
        else:
            st.warning("🆓 Free recommended: Ideal for casual listeners.")

    # --- 6. Data Insights ---
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.checkbox("View Training Data"):
            st.dataframe(df.tail(10))

    with col2:
        if st.checkbox("Show Model Info"):
            st.write("**Genres recognized:**")
            st.write(", ".join(le_genre.classes_))
            st.write("**Features used:** Age, Gender, Usage Hours, Genre")

else:
    st.warning("⚠️ Please upload a CSV file to continue.")
