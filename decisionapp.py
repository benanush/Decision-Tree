import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- Page Config ---
st.set_page_config(page_title="Music Subscription Predictor", layout="wide")

st.title(" Music Subscription Plan Predictor")
st.write("This app predicts the best **Subscription Plan** for a user based on their demographics and music taste.")

# --- 1. Load and Prepare Data ---
@st.cache_resource
def load_and_train():
    # Load the dataset
    df = pd.read_csv(r'data\C:\Users\benan\Documents\Data_Scientist\Streamlit\Movie_Interests_DecisionTree.csv')
    
    # Initialize LabelEncoder for the Genre column
    le_genre = LabelEncoder()
    df['Genre_Code'] = le_genre.fit_transform(df['Preferred_Genre'])
    
    # Prepare Features (X) and Target (y)
    X = df[['Age', 'Gender', 'Usage_Hours', 'Genre_Code']]
    y = df['Subscription_Plan']
    
    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X.values, y)
    
    return model, le_genre, df

model, le_genre, df = load_and_train()

# --- 2. Sidebar for User Input ---
st.sidebar.header("User Information")

age = st.sidebar.slider("Age", min_value=10, max_value=80, value=25)
gender_text = st.sidebar.selectbox("Gender", options=["Male", "Female"])
usage = st.sidebar.number_input("Daily Usage Hours", min_value=0, max_value=24, value=5)
genre_text = st.sidebar.selectbox("Preferred Genre", options=le_genre.classes_)

# --- 3. Pre-process Input for Model ---
# Convert inputs back to numerical format
gender_encoded = 1 if gender_text == "Male" else 0
genre_encoded = le_genre.transform([genre_text])[0]

# --- 4. Prediction Logic ---
if st.sidebar.button("Predict Plan"):
    # Create the feature array for prediction
    # Input order: [Age, Gender, Usage_Hours, Genre_Code]
    prediction = model.predict([[age, gender_encoded, usage, genre_encoded]])
    
    # UI Display
    st.subheader(f"Recommended Plan: **{prediction[0]}**")
    
    # Add custom messages based on prediction
    if prediction[0] == "Premium":
        st.success("💎 **Premium recommended:** Perfect for power users who want the best experience!")
    elif prediction[0] == "Basic":
        st.info("⭐ **Basic recommended:** A great balance of features and value.")
    else:
        st.warning("🆓 **Free recommended:** Ideal for casual listeners.")

# --- 5. Data Insights ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.checkbox("View Training Data"):
        st.write("Last 10 entries of the dataset:")
        st.dataframe(df.tail(10))

with col2:
    if st.checkbox("Show Model Logic"):
        st.write("The model classifies users based on these categories:")
        st.write(f"**Genres recognized:** {', '.join(le_genre.classes_)}")

        st.write("**Features used:** Age, Gender, Usage Hours, Genre Preference")
