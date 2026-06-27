# 🎓 Student Marks Predictor

An AI-powered web app that predicts student academic performance using Machine Learning.

## Live Demo
🔗 [Click here to try the app](https://student-marks-predictor.streamlit.app)

## What it does
- Enter student details like study hours, attendance, previous score
- ML model predicts the final exam score
- Shows grade (A+, A, B, C, F) and Pass/Fail result
- Displays which factors affect the score the most

## Tech Stack
- Python
- Scikit-learn (Random Forest Regressor)
- Pandas & Numpy
- Streamlit

## Features
- Interactive sliders for all inputs
- Real-time score prediction
- Grade and Pass/Fail badge
- Feature importance visualization

## Model Details
- Algorithm: Random Forest Regressor
- Training samples: 1000
- R² Score: ~0.98
- Features used: study hours, attendance, previous score, sleep hours, extra activities

## How to run locally
pip install -r requirements.txt
streamlit run app.py

## What I learned
- Data preprocessing and feature engineering
- Training and evaluating ML regression models
- Building interactive web apps with Streamlit
- Deploying ML apps to the cloud