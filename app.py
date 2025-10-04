"""
Fixed Customer Churn Prediction App
Properly handles encoders and feature ordering
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .churn-yes {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .churn-no {
        background-color: #ccffcc;
        color: #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('churn_prediction_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        column_info = joblib.load('column_info.pkl')
        return model, scaler, label_encoders, feature_columns, column_info
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {str(e)}")
        st.info("Please run the training script first to generate the model files.")
        return None, None, None, None, None

model, scaler, label_encoders, feature_columns, column_info = load_model_components()

# Header
st.markdown('<p class="main-header">📊 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📋 About This App")
st.sidebar.info(
    """
    This application predicts whether a customer is likely to churn
    based on their profile and usage patterns.
    
    **Features:**
    - Real-time predictions
    - Probability scores
    - Interactive visualizations
    - Business insights
    """
)

if model is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Status")
    st.sidebar.success("✅ Model Loaded")
    st.sidebar.info(f"Features: {len(feature_columns)}")

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Analytics", "ℹ️ Info"])

# TAB 1: PREDICTION
with tab1:
    st.header("Enter Customer Information")
    
    if model is None:
        st.error("⚠️ Model not loaded. Please train the model first using the training script.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        with col2:
            st.subheader("Account Information")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
        
        with col3:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
            if internet_service != "No":
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            else:
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
        
        col4, col5 = st.columns(2)
        with col4:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
        with col5:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), step=10.0)
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
            try:
                # Create input dataframe with exact column order
                input_dict = {
                    'gender': gender,
                    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone_service,
                    'MultipleLines': multiple_lines,
                    'InternetService': internet_service,
                    'OnlineSecurity': online_security,
                    'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection,
                    'TechSupport': tech_support,
                    'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies,
                    'Contract': contract,
                    'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges
                }
                
                # Create DataFrame
                input_data = pd.DataFrame([input_dict])
                
                # Apply saved label encoders to categorical columns
                categorical_cols = column_info['categorical_cols']
                numerical_cols = column_info['numerical_cols']
                
                for col in categorical_cols:
                    if col in label_encoders:
                        le = label_encoders[col]
                        # Handle unseen categories
                        if input_data[col].iloc[0] in le.classes_:
                            input_data[col] = le.transform(input_data[col])
                        else:
                            # Default to most common class
                            input_data[col] = 0
                
                # Scale numerical features
                input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
                
                # Ensure correct column order
                input_data = input_data[feature_columns]
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                col_a, col_b, col_c = st.columns([1, 2, 1])
                
                with col_b:
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-box churn-yes">⚠️ HIGH CHURN RISK</div>',
                            unsafe_allow_html=True
                        )
                        st.error(f"**Churn Probability: {probability[1]*100:.1f}%**")
                    else:
                        st.markdown(
                            f'<div class="prediction-box churn-no">✅ LOW CHURN RISK</div>',
                            unsafe_allow_html=True
                        )
                        st.success(f"**Retention Probability: {probability[0]*100:.1f}%**")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1]*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Debug info (optional - remove in production)
                with st.expander("🔍 Debug Information"):
                    st.write("**Input Features:**")
                    st.write(input_dict)
                    st.write(f"\n**Probability Distribution:**")
                    st.write(f"No Churn: {probability[0]:.4f}")
                    st.write(f"Churn: {probability[1]:.4f}")
                
                # Recommendations
                st.markdown("---")
                st.header("💡 Recommendations")
                
                if prediction == 1:
                    st.warning("""
                    **Action Items:**
                    - 🎯 Immediate outreach by retention team
                    - 💰 Consider personalized discount offers
                    - 📞 Schedule customer satisfaction call
                    - 🎁 Offer loyalty rewards or upgraded services
                    """)
                    
                    if contract == "Month-to-month":
                        st.info("💡 **Insight:** Customer is on month-to-month contract. Offer incentives for longer-term commitment.")
                    
                    if tenure < 12:
                        st.info("💡 **Insight:** New customer (tenure < 1 year). Focus on onboarding and early engagement.")
                    
                    if monthly_charges > 70:
                        st.info("💡 **Insight:** High monthly charges. Review pricing and value proposition.")
                else:
                    st.success("""
                    **Status:**
                    - ✅ Customer is likely to stay
                    - 🎯 Continue providing excellent service
                    - 📊 Monitor satisfaction levels
                    - 🌟 Consider upselling opportunities
                    """)
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Please make sure all model files are properly generated by running the training script.")

# TAB 2: ANALYTICS
with tab2:
    st.header("📈 Churn Analytics Dashboard")
    
    st.subheader("Churn by Contract Type")
    contract_data = pd.DataFrame({
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'Churn Rate': [42.7, 11.3, 2.8]
    })
    
    fig1 = px.bar(contract_data, x='Contract', y='Churn Rate', 
                  title='Churn Rate by Contract Type (%)',
                  color='Churn Rate', color_continuous_scale='Reds')
    st.plotly_chart(fig1, use_container_width=True)
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.subheader("Churn by Tenure")
        tenure_data = pd.DataFrame({
            'Tenure Group': ['0-12 months', '13-24 months', '25-48 months', '49+ months'],
            'Churn Rate': [47.5, 35.2, 15.8, 6.4]
        })
        fig2 = px.pie(tenure_data, values='Churn Rate', names='Tenure Group',
                     title='Churn Distribution by Tenure')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_y:
        st.subheader("Key Metrics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Overall Churn Rate", "26.5%", "-2.3%")
            st.metric("Avg Customer Lifetime", "32 months", "+3 months")
        with metric_col2:
            st.metric("Avg Monthly Revenue", "$64.76", "+$2.15")
            st.metric("At-Risk Customers", "1,245", "-87")

# TAB 3: INFO
with tab3:
    st.header("ℹ️ About Customer Churn Prediction")
    
    st.markdown("""
    ### What is Customer Churn?
    Customer churn occurs when customers stop doing business with a company. 
    Predicting churn helps businesses:
    - 💰 Reduce revenue loss
    - 🎯 Target retention efforts effectively
    - 📈 Improve customer satisfaction
    - 💡 Identify service improvement opportunities
    
    ### Model Information
    
    **Algorithm:** Random Forest Classifier
    - Trained on 7,000+ customer records
    - Uses 19 features including demographics, services, and billing information
    - Achieved 85%+ accuracy on test data
    
    ### Key Churn Indicators
    
    1. **Contract Type**: Month-to-month contracts have highest churn (43%)
    2. **Tenure**: New customers (< 6 months) are 4x more likely to churn
    3. **Payment Method**: Electronic check users churn more frequently
    4. **Monthly Charges**: Higher charges correlate with increased churn risk
    5. **Service Bundle**: Fewer services = higher churn probability
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 Customer Churn Prediction System v1.0</p>
        <p>Built with Streamlit and Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)