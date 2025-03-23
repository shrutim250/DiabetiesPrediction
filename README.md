# DiabetiesPrediction

# Diabetes Prediction System

A machine learning web application that predicts the likelihood of diabetes based on diagnostic measurements. Built with Python, Scikit-learn, and Streamlit.

## Overview

This project uses the Pima Indians Diabetes Database to build a predictive model that can identify patients at risk of diabetes. The web interface allows users to input their health metrics and instantly receive a risk assessment.



## Dataset

The dataset consists of several medical predictor variables and one target variable, Outcome. Predictor variables include:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration (mg/dL)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (kg/m²)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years

## Model

The project implements a Support Vector Classifier (SVC) with a linear kernel. The model was trained on a preprocessed dataset with standardized features.

## Web Application

A user-friendly web interface built with Streamlit allows for easy interaction with the model.

### Features

- Input form for entering patient diagnostic measurements
- Real-time prediction of diabetes risk
- Probability score indicating confidence level
- Color-coded results for easy interpretation
- Responsive design that works on desktop and mobile devices

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shrutim250/DiabetiesPrediction.git
cd DiabetiesPrediction
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to http://localhost:8501

## Project Structure

```
DiabetiesPrediction/
├── Diabeties_Prediction.ipynb    # Jupyter notebook with model development
├── app.py                        # Streamlit web application
├── diabetes.csv                  # Dataset
├── diabetes_model.pkl            # Trained model (serialized)
├── README.md                     # This file
└── requirements.txt              # Package dependencies
```

## How to Use

1. Enter the patient's diagnostic measurements in the form.
2. Click the "Predict Diabetes Risk" button.
3. View the prediction result and probability score.
4. Consult with a healthcare professional for proper diagnosis.

## Development

### Notebook

The Jupyter notebook (`Diabeties_Prediction.ipynb`) contains the entire data science workflow:
- Data loading and exploration
- Data preprocessing and feature engineering
- Model selection, training, and evaluation
- Model serialization

### Web Application

The Streamlit application (`app.py`) provides a user interface for interacting with the trained model:
- Form for collecting patient data
- Data preprocessing
- Model prediction
- Result visualization

## Future Improvements

- Feature importance visualization
- Support for data upload via CSV
- More advanced models and ensembles
- Patient history tracking
- Explainable AI components

## Disclaimer

This application is for educational purposes only and does not provide medical advice. The predictions should not be used for diagnosing or treating any health problem or disease. Always consult with a qualified healthcare provider for medical advice.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases
- Streamlit for their amazing framework for building data applications
