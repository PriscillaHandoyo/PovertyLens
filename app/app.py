import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
def load_data():
    data = pd.read_csv('../datasets/poverty_dataset_clean.csv')
    return data

# Load the model 
def load_model():
    try:
        model = joblib.load('../models/random_forest_model.pkl')
        feature_names = joblib.load('../models/feature_names.pkl')
        return model, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please run the model training notebook first.")
        return None, None

# Predict poverty risk
def predict_poverty(model, data, feature_names):
    # Select only the features the model was trained on
    model_data = data[feature_names]
    predict = model.predict(model_data)
    probability = model.predict_proba(model_data)[0][1]
    return predict[0], probability

# Streamlit app
def main():
    st.title("PovertyLens")
    st.write("Poverty Risk Prediction App based on Socioeconomic Indicators")

    # Load data and model
    data = load_data()
    model_result = load_model()
    
    # Check if model loaded successfully
    if model_result[0] is None:
        st.stop()
    
    model, feature_names = model_result

    # Sidebar for user input
    st.sidebar.header("User Input Features")
    input_data = {}
    
    model_features = ['Literacy Rate', 'Unemployment Rate', 'GDP per Capita',
                     'Infant Mortality Rate', 'Health Expenditure']

    for col in model_features:
        label_map = {
            'Literacy Rate': 'Literacy Rate (%)',
            'Unemployment Rate': 'Unemployment Rate (%)',
            'GDP per Capita': 'GDP per Capita (PPP)',
            'Infant Mortality Rate': 'Infant Mortality Rate (per 1000)',
            'Health Expenditure': 'Health Expenditure (PPP)'
        }
        
        label = label_map.get(col, col.replace('_', ' ').title())
        
        val = st.sidebar.slider(
            label=label,
            min_value=float(data[col].min()),
            max_value=float(data[col].max()),
            value=float(data[col].median()),
            help=f"Range: {data[col].min():.1f} - {data[col].max():.1f}"
        )
        input_data[col] = val
    
    input_df = pd.DataFrame([input_data])

    if st.sidebar.button("Predict Poverty Risk"):
        try:
            pred, proba = predict_poverty(model, input_df, feature_names)
            
            st.subheader("Prediction Result")
            
            # Display result with color coding
            if pred == 1:
                st.error(f"🚨 **High Poverty Risk** (Probability: {proba:.2%})")
            else:
                st.success(f"✅ **Low Poverty Risk** (Probability: {proba:.2%})")
            
            # Show input values
            st.subheader("Input Values Used")
            for feature, value in input_data.items():
                st.write(f"**{feature.replace('_', ' ').title()}:** {value:.2f}")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Feature importance
    st.subheader("Feature Importance")
    st.write("These factors have the most influence on poverty risk prediction:")
    
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': [name.replace('_', ' ').title() for name in feature_names],
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
        ax.set_title("Feature Importance in Poverty Risk Prediction")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

    # Dataset overview
    st.subheader("Dataset Overview")
    st.write(f"Dataset contains {len(data)} records from {data['Country Name'].nunique()} countries")
    st.write("Sample data:")
    st.dataframe(data.head())

if __name__ == "__main__":
    main()