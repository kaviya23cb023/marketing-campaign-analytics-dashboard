import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Set page config
st.set_page_config(page_title="Marketing Dashboard", layout="wide", page_icon="📈")

# Apply custom CSS animations and styles
st.markdown("""
<style>
/* Base Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Apply fade in to the entire main block */
.main .block-container {
    animation: fadeIn 0.8s ease-out;
}

/* Styled KPI Cards */
.metric-card {
    background-color: rgba(128, 128, 128, 0.05);
    border-radius: 12px;
    padding: 20px 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    text-align: center;
    border-top: 4px solid #4CAF50;
    margin-bottom: 20px;
    animation: slideIn 0.5s ease-out;
}

.metric-card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 8px 15px rgba(0,0,0,0.1);
}

.metric-title {
    font-size: 14px;
    color: gray;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    font-weight: 600;
}

.metric-val {
    font-size: 26px;
    font-weight: 700;
}

.text-green { color: #4CAF50; }
.text-red { color: #F44336; }
.text-blue { color: #2196F3; }
.text-neutral { color: inherit; }

</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Marketing Campaign Performance Dashboard")
st.markdown("Transforming business data into real-world decision support systems with intelligent analytics.")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('marketing_campaign_dataset_5000.csv')

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Handling missing values
df = df.dropna()

# Global Model Prep
features = ['Age', 'Gender', 'Income', 'Campaign_Type', 'Campaign_Cost', 'Clicked_Ad']
X = df[features].copy()
y = df['Purchased']
X_encoded = pd.get_dummies(X, columns=['Gender', 'Campaign_Type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train baseline Random Forest to extract Feature Information
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train_scaled, y_train)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard & UI", "Model Evaluation", "Real-Time Prediction"])

total_customers = len(df)
st.sidebar.markdown('---')
st.sidebar.write(f"**Total Customers Analysed**: {total_customers:,}")

if page == "Dashboard & UI":
    st.header("📊 Business Overview & Analytics")
    st.caption("Gain a comprehensive top-down view of customer conversion and campaign health.")
    
    # Overview Metrics Calculations
    total_revenue = df['Revenue'].sum()
    total_cost = df['Campaign_Cost'].sum()
    total_purchases = df['Purchased'].sum()
    conversion_rate = (total_purchases / total_customers) * 100
    roi = ((total_revenue - total_cost) / total_cost) * 100
    
    # Custom HTML KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #2196F3;">
            <div class="metric-title">Total Customers</div>
            <div class="metric-val text-blue">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #FF9800;">
            <div class="metric-title">Total Cost</div>
            <div class="metric-val">${total_cost:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #4CAF50;">
            <div class="metric-title">Total Revenue</div>
            <div class="metric-val text-green">${total_revenue:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #9C27B0;">
            <div class="metric-title">Conversion Rate</div>
            <div class="metric-val">{conversion_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    roi_color = "text-green" if roi >= 0 else "text-red"
    roi_border = "#4CAF50" if roi >= 0 else "#F44336"
    
    with col5:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: {roi_border};">
            <div class="metric-title">Overall ROI</div>
            <div class="metric-val {roi_color}">{roi:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()

    # Insights Section
    st.subheader("💡 Automated Business Insights")
    st.info("The logic engine automatically crawls feature distributions and ranks performance correlations.")
    
    importances = rf_base.feature_importances_
    feature_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    top_feature1 = feature_df.iloc[0]['Feature']
    top_feature2 = feature_df.iloc[1]['Feature']
    
    roi_df = df.groupby('Campaign_Type').agg(Cost=('Campaign_Cost', 'sum'), Revenue=('Revenue', 'sum')).reset_index()
    roi_df['ROI (%)'] = ((roi_df['Revenue'] - roi_df['Cost']) / roi_df['Cost']) * 100
    best_campaign = roi_df.loc[roi_df['ROI (%)'].idxmax()]['Campaign_Type']
    
    in1, in2, in3 = st.columns(3)
    in1.success(f"**Top Influencing Features**\n\nThe features '{top_feature1}' and '{top_feature2}' definitively act as top drivers for customer purchasing.")
    in2.success(f"**Best Campaign Type**\n\nBased on mathematics mathematically, '{best_campaign}' drives the highest rate of Return on Investment.")
    
    if roi >= 0:
        in3.success(f"**ROI Analysis**\n\nOverall ROI sits healthily at **{roi:.2f}%**. Keep monitoring specific costs.")
    else:
        in3.error(f"**ROI Analysis**\n\nOverall ROI is currently at **{roi:.2f}%**. Re-evaluate heavily invested channels.")
    
    st.divider()
    
    # Visual Analytics
    st.subheader("📉 Visual Analytics & Distributions")
    sns.set_theme(style="whitegrid", context="talk") # Increase general font readability natively

    cols1, cols2 = st.columns(2)
    with cols1:
        st.markdown("**1. Campaign Type vs Purchases**")
        st.caption("Visually compares the raw unit sales generated across channels.")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=df, x='Campaign_Type', y='Purchased', estimator=sum, errorbar=None, palette='viridis', hue='Campaign_Type', legend=False, ax=ax)
        ax.set_title("Total Purchases Extracted by Medium", fontsize=15, pad=10)
        ax.set_xlabel("Campaign Medium", fontsize=13)
        ax.set_ylabel("Total Purchases", fontsize=13)
        st.pyplot(fig)
        
    with cols2:
        st.markdown("**2. Purchase Distribution**")
        st.caption("Highlights the split between successful and unsuccessful targeting.")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        purchases_counts = df['Purchased'].value_counts()
        ax2.pie(purchases_counts, labels=['No Purchase (0)', 'Purchase (1)'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], explode=(0.05, 0), shadow=True, startangle=140)
        ax2.set_title("Overall Target Hit Distribution", fontsize=15, pad=10)
        st.pyplot(fig2)
        
    st.markdown("**3. Campaign Cost vs Revenue (Successful Purchases Only)**")
    st.caption("Identifies diminishing returns and linear clusters associated with marketing spend.")
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    sns.scatterplot(data=df[df['Purchased'] == 1], x='Campaign_Cost', y='Revenue', hue='Campaign_Type', palette='deep', s=80, alpha=0.8, ax=ax3)
    ax3.set_title("Cost vs Collected Revenue for Conversions", fontsize=15, pad=10)
    ax3.set_xlabel("Campaign Spend ($)", fontsize=13)
    ax3.set_ylabel("Succeeding Revenue ($)", fontsize=13)
    st.pyplot(fig3)

elif page == "Model Evaluation":
    st.header("⚙️ Machine Learning Evaluation Suite")
    st.caption("Strict comparative metrics calculating precision against the baseline parameters.")
    
    st.info("💡 Selecting the Random Forest Classifier generally maps to stronger accuracy on non-linear distributions.")
    
    model_choice = st.selectbox("Select Model to Evaluate", ["Logistic Regression", "Random Forest Classifier"])
    
    if st.button("Run Strict Evaluation"):
        with st.spinner("Crunching vectors and plotting matrices..."):
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            st.markdown("### Accuracy Metrics")
            m1, m2, m3 = st.columns(3)
            
            # Using Metric Cards for ML results as well
            with m1:
                st.markdown(f"""
                <div class="metric-card" style="border-top-color: #2196F3;">
                    <div class="metric-title">Accuracy</div>
                    <div class="metric-val">{acc:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card" style="border-top-color: #9C27B0;">
                    <div class="metric-title">Precision</div>
                    <div class="metric-val">{precision:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card" style="border-top-color: #FF9800;">
                    <div class="metric-title">Recall</div>
                    <div class="metric-val">{recall:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            col_cm, _ = st.columns(2)
            with col_cm:
                st.markdown("### Model Confusion Matrix")
                st.caption("Mapping actual vs predicted results mathematically.")
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.set_theme(style="white") # Reset style for heatmap clarity
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'], annot_kws={"size": 14})
                st.pyplot(fig_cm)

elif page == "Real-Time Prediction":
    st.header("🔮 Customer Prediction Algorithm")
    st.caption("Operate the trained engine by entering abstract customer configurations.")
    
    with st.form("predict_form"):
        st.subheader("Customer DNA Input Form")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Household Income ($)", min_value=10000, max_value=200000, value=50000, step=1000)
            gender = st.selectbox("Designated Gender", ["Male", "Female"])
        
        with col_f2:
            campaign = st.selectbox("Marketed Campaign Type", ["Email", "SMS", "TV", "Social Media"])
            cost = st.number_input("Individual Campaign Cost ($)", min_value=0, max_value=10000, value=500)
            clicked = st.selectbox("Did They Click The Ad?", ["No", "Yes"])
            
        submit = st.form_submit_button("Run Predictive Outcome")
        
        if submit:
            with st.spinner("Processing input through neural ensemble layers..."):
                clicked_int = 1 if clicked == "Yes" else 0
                
                input_df = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender],
                    'Income': [income],
                    'Campaign_Type': [campaign],
                    'Campaign_Cost': [cost],
                    'Clicked_Ad': [clicked_int]
                })
                
                input_encoded = pd.get_dummies(input_df)
                input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
                input_scaled = scaler.transform(input_encoded)
                
                prediction = rf_base.predict(input_scaled)[0]
                probability = rf_base.predict_proba(input_scaled)[0][1]
                
                st.divider()
                st.subheader("Machine Learning Decision")
                if prediction == 1:
                    st.success(f"🎯 **HIGH PROBABILITY FLAG:** The customer configuration **WILL LIKELY PURCHASE**! (Probability: {probability:.2%})")
                else:
                    st.error(f"🛑 **LOW PROBABILITY FLAG:** The customer will theoretically **NOT PURCHASE**. (Probability: {probability:.2%})")
