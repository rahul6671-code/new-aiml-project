import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

st.title("🎯 Employee Attrition Prediction Dashboard")
st.markdown("### Business Problem: Reduce Employee Exit")

# 1. STREAMLIT UPLOAD INTERFACE
uploaded_file = st.sidebar.file_uploader("Upload your Employee Dataset (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 2. MODELING (Logistic Regression)
    try:
        features = ['Satisfaction', 'Evaluation', 'Projects', 'Monthly_Hours', 'Tenure', 'Salary_Level']
        X = df[features]
        y = df['Attrition']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        # Predictions
        df['Risk_Score'] = (model.predict_proba(X_scaled)[:, 1] * 100).round(2)
        df['Status'] = df['Risk_Score'].apply(lambda x: 'High Risk' if x > 70 else ('Medium Risk' if x > 40 else 'Low Risk'))

        # 3. DASHBOARD LAYOUT
        col1, col2 = st.columns(2)

        with col1:
            # Chart 1: Bar Chart
            fig1 = px.bar(df, x='Employee_ID', y='Risk_Score', color='Status',
                         title='Individual Employee Attrition Risk (%)',
                         color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Chart 2: Feature Importance
            importance = np.abs(model.coef_[0])
            feat_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
            fig2 = px.pie(feat_importance, values='Importance', names='Feature', 
                         title='Factors Driving Exits', hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Scatter Plot
        fig3 = px.scatter(df, x='Satisfaction', y='Risk_Score', size='Monthly_Hours', color='Status',
                         title='Correlation: Satisfaction vs Risk (Size = Monthly Hours)')
        st.plotly_chart(fig3, use_container_width=True)

        # 4. STRATEGIC SUMMARY
        st.markdown("---")
        high_risk_count = len(df[df['Status'] == 'High Risk'])
        st.subheader("📢 Summary & Retention Strategy")
        
        c1, c2 = st.columns(2)
        c1.metric("Total Employees", len(df))
        c2.metric("High Risk Employees", high_risk_count, delta_color="inverse")

        if high_risk_count > 0:
            st.warning("*Recommendation:* Focus on work-life balance for high-risk Gen Z employees. Implement flexible hours.")
        else:
            st.success("Overall attrition risk is low. Maintain current engagement.")

    except Exception as e:
        st.error(f"Error: Please ensure your file has the correct column names: {features}")

else:
    st.info("Waiting for file upload in the sidebar...")
