import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Custom CSS load karo
with open("style.css", "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# -------------------------------
# Title aur description
# -------------------------------
st.title("IPL Score Predictor 🏏")
st.write("Current match situation daalo aur final score predict karo! (First innings ke liye)")

# -------------------------------
# Model load karo
# -------------------------------
try:
    model = joblib.load('ipl_score_predictor_model.pkl')
    st.success("Model successfully loaded!")
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

# -------------------------------
# Real teams aur venues from your dataset
# -------------------------------
batting_teams = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

bowling_teams = batting_teams.copy()  # Same teams bowling ke liye

venues = [
    "Narendra Modi Stadium (Ahmedabad, Gujarat)",
    "Eden Gardens (Kolkata, West Bengal)",
    "Wankhede Stadium (Mumbai, Maharashtra)",
    "M. A. Chidambaram Stadium (Chennai, Tamil Nadu)",
    "M. Chinnaswamy Stadium (Bengaluru, Karnataka)",
    "Salt Lake Stadium (Kolkata, West Bengal)",
    "Jawaharlal Nehru Stadium (New Delhi)",
    "Feroz Shah Kotla Ground (New Delhi)",
    "Rajiv Gandhi International Cricket Stadium (Hyderabad, Telangana)",
    "Sardar Patel Stadium (Ahmedabad, Gujarat)",
    "Punjab Cricket Association Stadium (Mohali, Punjab)",
    "Sawai Mansingh Stadium (Jaipur, Rajasthan)",
    "M. A. Chidambaram Stadium (Chennai, Tamil Nadu)",
]

# -------------------------------
# User inputs
# -------------------------------
batting_team = st.selectbox("Batting Team", batting_teams)
bowling_team = st.selectbox("Bowling Team", bowling_teams)
venue = st.selectbox("Venue", venues)

current_runs = st.slider("Current Runs", 0, 250, 80)          # Max IPL mein 250+ bhi ja sakta hai
current_wickets = st.slider("Wickets Fallen", 0, 9, 2)
over = st.slider("Current Over", 5.0, 19.5, 10.0, step=0.1)

# Derived features
balls_remaining = int((20 - over) * 6)
current_run_rate = current_runs / (over if over > 0 else 1)
over_number = int(over)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Predict Final Score"):
    # Input dictionary banao (numerical features)
    input_data = {
        'current_runs': current_runs,
        'current_wickets': current_wickets,
        'balls_remaining': balls_remaining,
        'current_run_rate': current_run_rate,
        'over_number': over_number,
    }

    # Empty dataframe banao
    input_df = pd.DataFrame([input_data])

    # One-hot columns add karo (exact prefix jo training mein use hua tha)
    # Note: Yeh prefix match karna zaroori hai – agar training mein 'batting_team_' tha to yahin use karo
    input_df[f'batting_team_{batting_team}'] = 1
    input_df[f'bowling_team_{bowling_team}'] = 1
    input_df[f'venue_{venue}'] = 1

    # Model ke original columns le lo aur missing ko 0 fill kar do
    original_columns = model.feature_names_in_   # XGBoost model se original feature names
    for col in original_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Columns order same kar do
    input_df = input_df[original_columns]

    # Predict
    predicted = model.predict(input_df)[0]

    st.success(f"**Predicted Final Score: {int(round(predicted))} runs**")
    st.info(f"(Average error ~ ±8 runs ho sakta hai based on your model)")