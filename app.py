import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Load pre-trained model
with open('model_forest.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Menambahkan gambar header untuk branding
header_image = Image.open('fc25eb6c76c940f3acb353da926134c7.jpg')
st.image(header_image, use_column_width=True)

# Page title and description
st.title('Heart Disease Prediction')
st.markdown("""
## Welcome to the **Heart Disease Prediction Application**
Take a proactive step towards better heart health. This app uses advanced machine learning algorithms to predict the likelihood of heart disease based on key health indicators. By simply inputting your personal health data, such as age, cholesterol levels, and blood pressure, you can receive an instant assessment. Whether you're a healthcare professional or an individual concerned about heart health, this tool provides a quick and informative way to understand potential risks. Start your journey towards a healthier heart today!
""")

# User input for the health indicators
def user_input_features():
    age = st.number_input('How old are You?', min_value=1, max_value=120, value=40)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fasting_bs = st.selectbox('Fasting Blood Sugar', (1.0, 0.0), help='1 is Yes/ 0 is No')
        oldpeak = st.number_input('Oldpeak (ST depression induced by exercise)', min_value=-2.600000, max_value=6.200000, value=1.0, help='ST depression induced by exercise relative to rest')
        max_hr = st.number_input('Max Heart Rate Achieved', min_value=60.0, max_value=202.0, value=150.0, help='Maximum heart rate')
        resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120,help='Blood pressure when resting')
        cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=603, value=200, help='Total cholesterol level')

    with col2:
        chest_pain_type = st.selectbox('Chest Pain Type', ('NAP', 'ASY', 'TA', 'ATA'))
        resting_ecg = st.selectbox('Resting ECG Results', ('Normal', 'LVH', 'ST'))
        sex = st.selectbox('Sex', ('M', 'F'))
        exercise_angina = st.selectbox('Exercise-Induced Angina', ('Y', 'N'))
        st_slope = st.selectbox('Slope of the Peak Exercise ST Segment', ('Up', 'Flat', 'Down'))

    # Convert user inputs to numerical format
    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction section
if st.button('Predict Heart Disease'):
    prediction = loaded_model.predict(input_df)
    prediction_proba = loaded_model.predict_proba(input_df)

    if prediction[0] == 1:
        st.warning('**Warning! You have a high risk of heart disease.**')
    else:
        st.success('**You are not at risk of heart disease.**')

    st.subheader('Prediction Probability')
    st.write(f"Risk of Heart Disease: {prediction_proba[0][1] * 100:.2f}%")
    st.write(f"No Risk: {prediction_proba[0][0] * 100:.2f}%")
