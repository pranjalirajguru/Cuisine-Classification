import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Cuisine Prediction", layout="wide")
st.title("🍽 Cuisine Classification 🍽")

# Load dataset and train
@st.cache_data
def load_and_train():
    df = pd.read_csv("Dataset .csv")

    # Clean cuisines
    df['Cuisines'] = df['Cuisines'].fillna('Unknown').apply(lambda x: x.split(',')[0].strip())
    counts = df['Cuisines'].value_counts()
    rare_cuisines = counts[counts < 5].index
    df['Cuisines'] = df['Cuisines'].replace(rare_cuisines, 'Other')

    # Encode target
    le_target = LabelEncoder()
    df['Cuisines'] = le_target.fit_transform(df['Cuisines'])

    # Feature columns
    features = [
        'Country Code', 'City', 'Longitude', 'Latitude', 'Average Cost for two',
        'Currency', 'Has Table booking', 'Has Online delivery',
        'Is delivering now', 'Price range', 'Aggregate rating', 'Votes'
    ]

    # Store encoders for categorical features
    encoders = {}
    for col in df[features].select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[features]
    y = df['Cuisines']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42,class_weight='balanced')
    model.fit(X_train, y_train)

    return model, le_target, encoders, features, df

model, le_target, encoders, features, df = load_and_train()

# Manual input form
st.subheader("Enter Restaurant Details:")

user_data = {}
for col in features:
    if col in encoders:  # categorical
        user_data[col] = st.selectbox(col, encoders[col].classes_)
    else:  # numeric
        user_data[col] = st.number_input(col, value=0.0)

if st.button("Predict Cuisine"):
    user_df = pd.DataFrame([user_data])

    # Encode categorical fields using stored encoders
    for col, le in encoders.items():
        user_df[col] = le.transform(user_df[col])

    prediction = model.predict(user_df)[0]
    cuisine_name = le_target.inverse_transform([prediction])[0]
    st.success(f"Predicted Cuisine: {cuisine_name}")

