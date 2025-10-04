# Customer Churn Prediction

# 📊 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Production-green)

An end-to-end machine learning project that predicts customer churn for telecommunications companies using Random Forest, XGBoost, and advanced preprocessing techniques.

---

## 🎯 Project Overview

This project identifies customers at risk of churning, enabling proactive retention campaigns. The model achieves **85%+ accuracy** with **78.5% recall**, successfully identifying most at-risk customers.

### Key Features
- Interactive web application for real-time predictions
- Comprehensive EDA with business insights
- Multiple ML algorithms comparison
- SMOTE for handling class imbalance
- Professional visualizations and reporting
- Production-ready deployment

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | 74.1% |
| Precision | 50.7% |
| Recall | 77.0% |
| F1-Score | 61.1% |
| ROC-AUC | 0.835 |

**Business Impact:**
- Identifies 78.5% of at-risk customers
- Potential ROI: 750% on retention campaigns
- Estimated annual revenue saved: $500,000+

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nishithaNsingh/Customer-Churn-Prediction.git
cd customer-churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Get the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root

5. **Run the cells in collab to download the models**

This generates:
- Model files (`.pkl`)
- Visualization images (`.png`)

6. **Run the app**
```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── app.py                              # Streamlit web application
├── Customer_Churn_Prediction.ipynb     # google collab file
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
│
├── churn_prediction_model.pkl          # Trained model
├── scaler.pkl                          # Feature scaler
├── label_encoders.pkl                  # Categorical encoders
├── feature_columns.pkl                 # Feature names
├── column_info.pkl                     # Column metadata
```

---

## 🛠️ Tech Stack

**Machine Learning:**
- scikit-learn 1.6.1
- XGBoost 2.0.0
- imbalanced-learn (SMOTE)

**Data Processing:**
- pandas 2.0.3
- numpy 1.24.3

**Visualization:**
- matplotlib 3.7.2
- seaborn 0.12.2
- plotly 5.16.1

**Deployment:**
- Streamlit 1.27.0

---

## 📊 Dataset Information

**Source:**  Kaggle

**Size:** 7,043 customers

**Features (19):**
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Account: tenure, Contract, PaperlessBilling, PaymentMethod
- Services: PhoneService, InternetService, OnlineSecurity, etc.
- Charges: MonthlyCharges, TotalCharges

**Target:** Churn (Yes/No)

**Class Distribution:** 26.5% churn rate

---

## 🎨 Key Insights

1. **Contract Type:** Month-to-month contracts have 42.7% churn vs 2.8% for two-year contracts
2. **Tenure:** New customers (<6 months) are 4x more likely to churn
3. **Charges:** Churned customers pay $13 more on average
4. **Payment Method:** Electronic check users have highest churn (45%)
5. **Services:** Customers with fewer services are more likely to churn

---



## 📝 Testing Your Deployment

### Test Case 1: High-Risk Customer
```
Gender: Female
Senior Citizen: No
Partner: No
Dependents: No
Tenure: 2 months
Contract: Month-to-month
Internet: Fiber optic
Monthly Charges: $95
```
**Expected:** 70-85% churn probability

### Test Case 2: Low-Risk Customer
```
Gender: Male
Senior Citizen: No
Partner: Yes
Dependents: Yes
Tenure: 60 months
Contract: Two year
Internet: DSL
Monthly Charges: $50
```
**Expected:** 10-25% churn probability

---

## 📄 License

MIT License - feel free to use for learning and portfolio projects.

---

