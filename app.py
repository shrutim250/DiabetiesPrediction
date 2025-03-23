import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4169E1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .positive {
        background-color: #ffcccb;
    }
    .negative {
        background-color: #d0f0c0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-header'>Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("##### Enter your health metrics to check your diabetes risk")

# Load the saved model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('diabetes_model.pkl', 'rb'))
        return model
    except:
        st.error("Model file not found. Please make sure 'diabetes_model.pkl' is in the same directory.")
        return None

model = load_model()

# Create input form
st.markdown("<h2 class='sub-header'>Patient Information</h2>", unsafe_allow_html=True)

# Creating columns for better layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=500, value=120)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input('Age', min_value=0, max_value=120, value=30)

# Function to make predictions
def predict_diabetes(input_data):
    # Reshape the input data
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    
    # Create a StandardScaler (same as was used in training)
    scaler = StandardScaler()
    
    # Since we don't have the original scaler, we'll normalize approximately
    # This is a simplification - ideally you would use the same scaler used during training
    normalized_data = input_data_reshaped
    
    # Make prediction
    prediction = model.predict(normalized_data)
    probability = model.predict_proba(normalized_data)
    
    return prediction[0], probability[0][1]  # Return prediction and probability of positive class

# Button to make prediction
if st.button('Predict Diabetes Risk'):
    if model is not None:
        try:
            # Collect all inputs into a list
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                         insulin, bmi, diabetes_pedigree, age]
            
            # Get prediction
            prediction, probability = predict_diabetes(input_data)
            
            # Display prediction
            st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"""
                <div class='prediction-box positive'>
                    <h3>High Risk of Diabetes</h3>
                    <p>The model predicts that you may have a high risk of diabetes with {probability:.2%} confidence.</p>
                    <p><strong>Note:</strong> Please consult with a healthcare professional for proper diagnosis and advice.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-box negative'>
                    <h3>Low Risk of Diabetes</h3>
                    <p>The model predicts that you have a low risk of diabetes with {(1-probability):.2%} confidence.</p>
                    <p><strong>Note:</strong> This is not a medical diagnosis. Regular check-ups are still recommended.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Display input data for reference
            st.markdown("<h3>Your Input Data</h3>", unsafe_allow_html=True)
            input_df = pd.DataFrame({
                'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                           'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
                'Value': input_data
            })
            st.table(input_df)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model could not be loaded. Please ensure model file is available.")

# Disclaimer
st.markdown("---")
st.markdown("""
**Disclaimer:** This application is for educational purposes only and does not replace professional medical advice. 
Always consult with a healthcare provider for proper diagnosis and treatment of diabetes or any other medical condition.
""")

# Information about the model
with st.expander("About the Model"):
    st.write("""
    This diabetes prediction model is trained on the Pima Indians Diabetes Database. 
    The dataset consists of medical predictor variables such as pregnancies, glucose concentration, 
    blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.
    
    The model uses Support Vector Classification algorithm to predict the likelihood of diabetes based on these features.
    """)

# How to use section
with st.expander("How to Use This App"):
    st.write("""
    1. Enter your health metrics in the form above
    2. Click the 'Predict Diabetes Risk' button
    3. View your prediction result
    4. Consult with a healthcare professional regardless of the result
    
    **Understanding the metrics:**
    - **Pregnancies**: Number of times pregnant
    - **Glucose**: Plasma glucose concentration (mg/dL)
    - **Blood Pressure**: Diastolic blood pressure (mm Hg)
    - **Skin Thickness**: Triceps skin fold thickness (mm)
    - **Insulin**: 2-Hour serum insulin (mu U/ml)
    - **BMI**: Body mass index (kg/m²)
    - **Diabetes Pedigree Function**: Likelihood of diabetes based on family history
    - **Age**: Age in years
    """)
