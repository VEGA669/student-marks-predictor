import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f0f13; }
    
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        text-align: center;
    }
    
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
    }
    
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: #1e1e2e;
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
        display: block;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .section-card {
        background: #1e1e2e;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
    }
    
    .section-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 20px;
    }

    .result-container {
        background: linear-gradient(135deg, #1a1a2e, #1e1e3e);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
    }
    
    .score-display {
        font-size: 5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 10px 0;
    }
    
    .grade-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .grade-a-plus { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
    .grade-a { background: rgba(34,197,94,0.1); color: #86efac; border: 1px solid rgba(34,197,94,0.2); }
    .grade-b { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }
    .grade-c { background: rgba(234,179,8,0.15); color: #fde047; border: 1px solid rgba(234,179,8,0.3); }
    .grade-f { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
    
    .pass-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 8px;
    }
    
    .pass { background: rgba(34,197,94,0.1); color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
    .fail { background: rgba(239,68,68,0.1); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

    .stSlider > div > div > div { background: #818cf8 !important; }
    
    div[data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    
    .stSelectbox label { color: #94a3b8 !important; font-size: 0.85rem !important; }

    .predict-btn > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
    }
    
    .insight-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .insight-label { color: #94a3b8; font-size: 0.85rem; }
    
    .insight-bar-container {
        flex: 1;
        margin: 0 16px;
        height: 4px;
        background: rgba(255,255,255,0.05);
        border-radius: 2px;
    }
    
    .insight-bar {
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, #6366f1, #c084fc);
    }
    
    .insight-pct { color: #818cf8; font-size: 0.8rem; font-weight: 600; min-width: 36px; text-align: right; }
    
    .stApp { background: #0f0f13 !important; }
    [data-testid="stAppViewContainer"] { background: #0f0f13 !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    section[data-testid="stSidebar"] { background: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    df = pd.read_csv('students.csv')
    X = df.drop('final_score', axis=1)
    y = df['final_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return model, r2, mae, X.columns.tolist()

model, r2, mae, features = train_model()

st.markdown("""
<div class="hero">
    <h1>EduPredict</h1>
    <p>AI-powered academic performance predictor using Random Forest ML</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <span class="metric-value">{r2:.2f}</span>
        <div class="metric-label">R² Accuracy</div>
    </div>
    <div class="metric-card">
        <span class="metric-value">{mae:.1f}</span>
        <div class="metric-label">Avg Error (marks)</div>
    </div>
    <div class="metric-card">
        <span class="metric-value">1000</span>
        <div class="metric-label">Training samples</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Student Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    study_hours = st.slider("Study hours per day", 1.0, 10.0, 5.0, 0.5)
    attendance = st.slider("Attendance %", 50.0, 100.0, 75.0, 1.0)
    previous_score = st.slider("Previous exam score", 40.0, 100.0, 70.0, 1.0)
with col2:
    sleep_hours = st.slider("Sleep hours per day", 4.0, 10.0, 7.0, 0.5)
    extra_activities = st.selectbox("Extra activities", [1, 0], format_func=lambda x: "Yes, participates" if x == 1 else "No, does not participate")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("Predict Performance", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if predict:
    input_data = pd.DataFrame([[study_hours, attendance, previous_score, sleep_hours, extra_activities]], columns=features)
    prediction = round(min(max(model.predict(input_data)[0], 0), 100), 1)

    if prediction >= 90:
        grade, grade_class = "A+  Exceptional", "grade-a-plus"
    elif prediction >= 80:
        grade, grade_class = "A  Excellent", "grade-a"
    elif prediction >= 70:
        grade, grade_class = "B  Good", "grade-b"
    elif prediction >= 60:
        grade, grade_class = "C  Average", "grade-c"
    else:
        grade, grade_class = "F  Needs Improvement", "grade-f"

    pass_class = "pass" if prediction >= 50 else "fail"
    pass_text = "PASS" if prediction >= 50 else "FAIL"

    st.markdown(f"""
    <div class="result-container">
        <p style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:2px; margin:0;">Predicted Score</p>
        <div class="score-display">{prediction}</div>
        <p style="color:#475569; font-size:0.8rem; margin:0 0 12px 0;">out of 100</p>
        <div class="grade-badge {grade_class}">{grade}</div><br>
        <span class="pass-badge {pass_class}">{pass_text}</span>
    </div>
    """, unsafe_allow_html=True)

    importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What drives this prediction</div>', unsafe_allow_html=True)
    
    feature_labels = {
        'study_hours': 'Study Hours',
        'attendance': 'Attendance',
        'previous_score': 'Previous Score',
        'sleep_hours': 'Sleep Hours',
        'extra_activities': 'Extra Activities'
    }
    
    for _, row in importance.iterrows():
        pct = round(row['Importance'] * 100, 1)
        label = feature_labels.get(row['Feature'], row['Feature'])
        st.markdown(f"""
        <div class="insight-item">
            <span class="insight-label">{label}</span>
            <div class="insight-bar-container">
                <div class="insight-bar" style="width:{pct}%"></div>
            </div>
            <span class="insight-pct">{pct}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
