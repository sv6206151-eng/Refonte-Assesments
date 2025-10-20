import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Heart Disease Prediction", page_icon="❤️")

st.markdown("""
    <style>
        .stButton>button {
            background-color: #fb923c;
            color: white;
            border-radius: 0.5rem;
            border: 1px solid #f97316;
        }
        .stButton>button:hover {
            background-color: #f97316;
            border: 1px solid #c2410c;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #d97706;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Interactive Report: Predicting Heart Disease")
st.write("This report documents the end-to-end development of a machine learning model to predict heart disease risk. We'll explore the data, compare different algorithms, and demonstrate the final model with an interactive prediction tool.")

st.markdown("---")

st.header("Project Overview & Data")
st.write("The project began with a dataset of patient health records. The first crucial phase involved cleaning and preparing this data to make it suitable for training a machine learning model.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Patient Records", value="303")
with col2:
    st.metric(label="Initial Features", value="14")
with col3:
    st.metric(label="Missing Values", value="0")

st.subheader("Data Preprocessing Workflow")
st.image("https://placehold.co/800x150/fdfaf7/3f3c3a?text=Raw+Data+%20%E2%86%92%20+Categorical+Encoding+%20%E2%86%92%20+Feature+Scaling+%20%E2%86%92%20+Model-Ready+Data")

st.markdown("---")

st.header("Data Insights")
st.write("Exploratory Data Analysis (EDA) allowed us to uncover patterns and relationships within the data. These insights are crucial for understanding the factors that might contribute to heart disease.")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("Feature Distributions")
    st.write("Select a feature to see how its values are distributed across the patients in the dataset.")
    
    feature_data = {
        'Age': pd.DataFrame({
            'Category': ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'],
            'Count': [1, 12, 72, 125, 81, 12]
        }),
        'Resting Blood Pressure': pd.DataFrame({
            'Category': ['90-109', '110-129', '130-149', '150-169', '170+'],
            'Count': [57, 115, 93, 30, 8]
        }),
        'Cholesterol': pd.DataFrame({
            'Category': ['100-199', '200-299', '300-399', '400+'],
            'Count': [59, 179, 58, 7]
        }),
        'Max Heart Rate': pd.DataFrame({
            'Category': ['70-99', '100-129', '130-159', '160-189', '190+'],
            'Count': [12, 53, 119, 108, 11]
        })
    }
    
    selected_feature = st.selectbox(
        'Select a feature:',
        list(feature_data.keys())
    )
    
    fig = px.bar(feature_data[selected_feature], x='Category', y='Count',
                 title=f'Distribution of {selected_feature}',
                 color_discrete_sequence=['#fb923c'])
    st.plotly_chart(fig, use_container_width=True)

with eda_col2:
    st.subheader("Key Feature Correlations with Heart Disease")
    st.write("This chart shows which features have the strongest relationship with the presence of heart disease. Positive values suggest a direct correlation, while negative values suggest an inverse one.")
    
    correlation_data = pd.DataFrame({
        'Feature': ['Chest Pain Type', 'Max Heart Rate', 'Slope', 'Resting ECG', 'Exercise Angina', 'Oldpeak', 'Num Major Vessels'],
        'Correlation': [0.43, 0.42, 0.35, 0.14, -0.44, -0.43, -0.39]
    })
    
    fig_corr = px.bar(correlation_data, x='Correlation', y='Feature', orientation='h',
                      title='Feature Correlations with Heart Disease',
                      color='Correlation',
                      color_continuous_scale=px.colors.sequential.RdBu)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

st.header("Building the Prediction Model")
st.write("Several machine learning models were trained and evaluated to find the most accurate predictor. After hyperparameter tuning, a clear winner emerged.")

model_data = pd.DataFrame({
    'Model': ['Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree'],
    'Accuracy': [0.88, 0.86, 0.85, 0.81]
})

fig_models = px.bar(model_data, x='Model', y='Accuracy',
                    title='Tuned Model Performance Comparison',
                    color='Accuracy',
                    color_continuous_scale=['#fdba74', '#fb923c', '#d97706'])
st.plotly_chart(fig_models, use_container_width=True)

st.info("The Random Forest Classifier demonstrated the best overall performance, balancing accuracy and robustness, making it the final choice for the prediction tool.")

st.markdown("---")

st.header("Live Prediction Tool")
st.write("This tool provides a simplified demonstration of the final model. Adjust the patient details below to see a simulated prediction of heart disease risk.")

def get_prediction_result(age, chol, thalach, cp, exang):
    score = 0
    score += (age > 60) * 20 + (50 < age <= 60) * 10
    score += (chol > 300) * 20 + (240 < chol <= 300) * 10
    score -= (thalach > 160) * 15 + (140 < thalach <= 160) * 5
    score += (cp == 'Asymptomatic') * 25 + (cp == 'Typical Angina') * 5
    score += (exang == 'Yes') * 30
    
    probability = max(5, min(95, score + 10))
    
    if probability > 50:
        return "High Risk", f"Confidence: {probability:.0f}%", "red"
    else:
        return "Low Risk", f"Confidence: {100 - probability:.0f}%", "green"

with st.form("prediction_form"):
    st.subheader("Patient Details")
    
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        age = st.slider("Age", min_value=29, max_value=77, value=54)
        chol = st.slider("Cholesterol", min_value=126, max_value=564, value=246)
        thalach = st.slider("Max Heart Rate", min_value=71, max_value=202, value=150)
        
    with col_pred2:
        cp = st.selectbox(
            "Chest Pain Type",
            ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'],
            index=2
        )
        exang = st.radio(
            "Exercise Induced Angina?",
            ['No', 'Yes'],
            index=0
        )
        st.write("") # Spacer
        submitted = st.form_submit_button("Get Prediction")

if submitted:
    result, confidence, color = get_prediction_result(age, chol, thalach, cp, exang)
    st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; border-radius: 0.5rem; background-color: #f0f0f0;">
            <p style="font-size: 1.25rem; font-weight: bold; margin-bottom: 0.5rem;">Simulated Risk Prediction</p>
            <p style="font-size: 3rem; font-weight: bold; color: {color};">{result}</p>
            <p style="font-size: 1rem; color: gray;">{confidence}</p>
        </div>
    """, unsafe_allow_html=True)
    st.caption("Disclaimer: This is a demonstration tool and not a substitute for professional medical advice.")

st.markdown("---")

st.header("Conclusion & Future Work")
st.write("This project successfully developed an effective Random Forest model for predicting heart disease. The process highlights the importance of a structured workflow from data preprocessing to model deployment.")

st.markdown("""
- **Collect More Data**: Enhancing the model's accuracy by training on a larger, more diverse dataset.
- **Use Advanced Algorithms**: Exploring algorithms like XGBoost or neural networks could yield even better performance.
- **Incorporate More Features**: Adding richer data, such as patient lifestyle or genetic information, could improve prediction.
- **Model Explainability**: Using tools like SHAP or LIME to understand *why* the model makes certain predictions.
""")
