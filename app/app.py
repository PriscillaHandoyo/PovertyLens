import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import pycountry 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
def load_data():
    try:
        # Try different possible paths
        possible_paths = [
            'poverty_dataset_clean.csv',
            '../datasets/poverty_dataset_clean.csv',  # This should match your notebook output
            'datasets/poverty_dataset_clean.csv',
            '../poverty_dataset_clean.csv',
            'data/poverty_dataset_clean.csv',
            '../data/poverty_dataset_clean.csv'
        ]
        
        for path in possible_paths:
            try:
                data = pd.read_csv(path)
                 # Add ISO3 column after loading
                def get_iso3(country_name):
                    try:
                        return pycountry.countries.lookup(country_name).alpha_3
                    except LookupError:
                        return None
                data['ISO3'] = data['Country Name'].apply(get_iso3)
                return data
            except FileNotFoundError:
                continue
        
        # If none of the paths work, show error
        st.error("Dataset file 'poverty_dataset_clean.csv' not found. Please ensure the file is in the correct directory.")
        return None
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load the model 
def load_model():
    try:
        # Try different possible paths for model files
        model_paths = [
            ('../models/random_forest_model.pkl', '../models/feature_names.pkl'),  # This should match your notebook output
            ('models/random_forest_model.pkl', 'models/feature_names.pkl'),
            ('random_forest_model.pkl', 'feature_names.pkl'),
            ('../random_forest_model.pkl', '../feature_names.pkl')
        ]
        
        for model_path, features_path in model_paths:
            try:
                model = joblib.load(model_path)
                feature_names = joblib.load(features_path)
                return model, feature_names
            except FileNotFoundError:
                continue
                
        st.error("Model files not found. Please run the model training notebook first.")
        return None, None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Predict poverty risk
def predict_poverty(model, data, feature_names):
    # Select only the features the model was trained on
    model_data = data[feature_names]
    predict = model.predict(model_data)
    probability = model.predict_proba(model_data)[0][1]
    return predict[0], probability
    
def main():
    st.title("PovertyLens")
    st.write("Poverty Risk Prediction App based on Socioeconomic Indicators")

    # Load data and model
    data = load_data()
    model_result = load_model()
    if data is None or model_result[0] is None:
        st.stop()
    model, feature_names = model_result

    # --- Country selection ---
    countries = sorted(data['Country Name'].unique())
    default_country = countries[0]
    selected_country = st.sidebar.selectbox("Select Country", countries, index=0)

    # --- Auto-fill inputs based on country ---
    country_row = data[data['Country Name'] == selected_country].iloc[0]
    if "reset" not in st.session_state:
        st.session_state.reset = False

    if st.sidebar.button("Reset to Defaults"):
        st.session_state.reset = True

    # Model features
    model_features = ['Literacy Rate', 'Unemployment Rate', 'GDP per Capita',
                      'Infant Mortality Rate', 'Health Expenditure']

    input_data = {}
    for col in model_features:
        label_map = {
            'Literacy Rate': 'Literacy Rate (%)',
            'Unemployment Rate': 'Unemployment Rate (%)',
            'GDP per Capita': 'GDP per Capita (PPP)',
            'Infant Mortality Rate': 'Infant Mortality Rate (per 1000)',
            'Health Expenditure': 'Health Expenditure (PPP)'
        }
        label = label_map.get(col, col.replace('_', ' ').title())
        default_val = float(country_row[col]) if not st.session_state.reset else float(data[col].median())
        val = st.sidebar.slider(
            label=label,
            min_value=0.0,
            max_value=float(data[col].max()),
            value=default_val,
            help=f"Range: 0.0 - {data[col].max():.1f}"
        )
        input_data[col] = val
    st.session_state.reset = False  # Reset only once

    input_df = pd.DataFrame([input_data])

    # --- Prediction ---
    if st.sidebar.button("Predict Poverty Risk"):
        pred, proba = predict_poverty(model, input_df, feature_names)
        st.subheader("Prediction Result")
        # Risk classification
        if proba >= 0.7:
            risk_label = "High"
            st.error(f"🚨 **High Poverty Risk** (Probability: {proba:.2%})")
        elif proba >= 0.4:
            risk_label = "Medium"
            st.warning(f"⚠️ **Medium Poverty Risk** (Probability: {proba:.2%})")
        else:
            risk_label = "Low"
            st.success(f"✅ **Low Poverty Risk** (Probability: {proba:.2%})")

        # --- Smart Insights ---
        st.markdown("### Smart Insights")
        insights = []
        for col in model_features:
            avg = data[col].mean()
            val = input_data[col]
            if col == "Unemployment Rate" and val > avg:
                insights.append(f"Unemployment rate ({val:.1f}%) is above global average ({avg:.1f}%).")
            if col == "GDP per Capita" and val < avg:
                insights.append(f"GDP per Capita ({val:.0f}) is below global average ({avg:.0f}).")
            if col == "Literacy Rate" and val < avg:
                insights.append(f"Literacy rate ({val:.1f}%) is below global average ({avg:.1f}%).")
            if col == "Infant Mortality Rate" and val > avg:
                insights.append(f"Infant mortality rate ({val:.1f}) is above global average ({avg:.1f}).")
            if col == "Health Expenditure" and val < avg:
                insights.append(f"Health expenditure ({val:.0f}) is below global average ({avg:.0f}).")
        if insights:
            for i in insights:
                st.markdown(f"- {i}")
        else:
            st.success("All selected indicators are close to or better than global averages.")

        # --- Country vs Global Comparison Chart ---
        st.markdown("### Country vs Global Averages")
        compare_df = pd.DataFrame({
            "Indicator": model_features,
            "Selected Country": [input_data[f] for f in model_features],
            "Global Average": [data[f].mean() for f in model_features]
        })
        fig = px.bar(compare_df, x="Indicator", y=["Selected Country", "Global Average"], barmode="group")
        st.plotly_chart(fig)

        # --- Map Visualization ---
        st.markdown("### Country Location")
        if "ISO3" in data.columns:
            iso_code = data[data['Country Name'] == selected_country]['ISO3'].values[0]
            map_df = pd.DataFrame({"iso_alpha": [iso_code], "Risk": [risk_label]})
            fig_map = px.choropleth(map_df, locations="iso_alpha", color="Risk",
                                color_discrete_map={"Low":"green","Medium":"orange","High":"red"},
                                locationmode="ISO-3", scope="world")
            st.plotly_chart(fig_map)
        else:
            st.info("Add an ISO3 column to your dataset for map visualization.")

        # --- Historical Trend ---
        if "Year" in data.columns:
            st.markdown("### Historical Trend")
            hist_df = data[data['Country Name'] == selected_country].sort_values("Year")
            fig_hist = px.line(hist_df, x="Year", y=model_features, title=f"{selected_country} Indicator Trends")
            st.plotly_chart(fig_hist)
        else:
            st.info("Add a 'Year' column to your dataset for historical trends.")

        # Input Summary
        st.markdown("### Summary of Your Input")
        for key, val in input_data.items():
            st.markdown(f"- **{key.replace('_', ' ').title()}**: {val}")

        # Smart Advice
        st.markdown("### Smart Advice")

        advice = []
        if input_data["Literacy Rate"] < 70:
            advice.append("**Improve education access**: Literacy rate is below 70%. Focus on primary and adult education initiatives.")
        if input_data["Unemployment Rate"] > 15:
            advice.append("**Tackle unemployment**: High unemployment can lead to systemic poverty. Explore job training and microfinance support.")
        if input_data["GDP per Capita"] < 5000:
            advice.append("**Boost economic activity**: Low GDP per capita indicates underdeveloped economy. Promote small business and investment.")
        if input_data["Infant Mortality Rate"] > 50:
            advice.append("**Invest in healthcare**: High infant mortality reflects poor health access. Improve clinics, maternity care, and sanitation.")
        if input_data["Health Expenditure"] < 200:
            advice.append("**Increase health investment**: Low spending on health may hinder long-term well-being.")

        if advice:
            for item in advice:
                st.markdown(f"- {item}")
        else:
            st.success("✅ All indicators are in a healthy range! Keep up the positive development trajectory.")

if __name__ == "__main__":
    main()