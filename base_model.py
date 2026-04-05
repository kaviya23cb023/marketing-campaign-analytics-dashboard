import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("--- 1. Load Dataset ---")
    df = pd.read_csv('marketing_campaign_dataset_5000.csv')
    print("Dataset loaded successfully.\n")
    df_original = df.copy() # Saving a copy for visualizations later

    print("--- 2. Basic Data Check ---")
    print("First 5 rows:")
    print(df.head(), "\n")
    
    print("Dataset Info:")
    df.info()
    print("\n")
    
    print("Missing Values:")
    print(df.isnull().sum(), "\n")

    print("--- 3. Simple Preprocessing ---")
    # Convert categorical columns into numeric format using pandas get_dummies
    df = pd.get_dummies(df, columns=['Gender', 'Campaign_Type'], drop_first=True)
    
    # Also ensuring boolean columns from get_dummies are ints natively (pandas 2.0+ behavior handling)
    df = df.astype({col: 'int' for col in df.select_dtypes(include=['bool']).columns})

    # Select important features
    features = ['Age', 'Income', 'Campaign_Cost', 'Clicked_Ad']
    
    # If we want to include the converted categorical columns, we add them to our features list.
    # The prompt explicitly listed Age, Income, Campaign_Cost, Clicked_Ad. We will use those.
    # But since we were asked to convert the others, let's include them as well as they are part of preprocessing.
    cat_features = [col for col in df.columns if 'Gender_' in col or 'Campaign_Type_' in col]
    selected_features = features + cat_features
    
    X = df[selected_features]
    y = df['Purchased']
    
    print(f"Selected Features: {list(X.columns)}\n")

    print("--- 4. Model Training & Comparison ---")
    # Split data into training and testing (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Training data size: {X_train.shape[0]}, Testing data size: {X_test.shape[0]}\n")

    # Implement Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Implement Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("--- 5. Model Evaluation ---")
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred)
    lr_rec = recall_score(y_test, lr_pred)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)

    print("Logistic Regression Results:")
    print(f"  Accuracy:  {lr_acc:.4f}")
    print(f"  Precision: {lr_prec:.4f}")
    print(f"  Recall:    {lr_rec:.4f}\n")

    print("Random Forest Results:")
    print(f"  Accuracy:  {rf_acc:.4f}")
    print(f"  Precision: {rf_prec:.4f}")
    print(f"  Recall:    {rf_rec:.4f}\n")

    # Best Model Selection
    if rf_acc > lr_acc:
        best_model_name = "Random Forest"
    else:
        best_model_name = "Logistic Regression"
    print(f"Best Model: {best_model_name}\n")

    print("--- 6. Feature Importance ---")
    # Using Random Forest to extract important features
    importances = rf_model.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False).head(5)
    
    print("Top 5 Important Features:")
    for idx, row in feature_df.iterrows():
        print(f" - {row['Feature']}: {row['Importance']:.4f}")
    print()

    print("--- 7. Business Metrics ---")
    total_customers = len(df)
    total_purchased = df['Purchased'].sum()
    conversion_rate = total_purchased / total_customers
    
    total_revenue = df['Revenue'].sum()
    total_cost = df['Campaign_Cost'].sum()
    
    if total_cost > 0:
        roi = (total_revenue - total_cost) / total_cost
    else:
        roi = 0.0

    print(f"Conversion Rate: {conversion_rate:.2%}")
    print(f"Total Revenue:   ${total_revenue:,.2f}")
    print(f"Total Cost:      ${total_cost:,.2f}")
    print(f"ROI:             {roi:.2%}")

    print("\n--- 8. Data Visualizations ---")
    print("Generating charts...")

    # Set visualization style
    sns.set_theme(style="whitegrid")

    # a) Campaign Type vs Purchases (Bar Chart)
    # Purpose: Identify best-performing campaign visually
    plt.figure(figsize=(8, 6))
    # We use df_original to keep the string labels for 'Campaign_Type'
    sns.barplot(data=df_original, x='Campaign_Type', y='Purchased', estimator=sum, errorbar=None, palette='viridis', hue='Campaign_Type', legend=False)
    plt.title("Total Purchases by Campaign Type", fontsize=14, fontweight='bold')
    plt.xlabel("Campaign Type", fontsize=12)
    plt.ylabel("Total Purchases", fontsize=12)
    # Comment: This bar chart displays the absolute number of successful purchases driven by each campaign type.

    # b) Purchase Distribution (Pie Chart)
    # Purpose: Understand overall conversion metrics globally
    plt.figure(figsize=(7, 7))
    purchase_counts = df_original['Purchased'].value_counts()
    plt.pie(purchase_counts, labels=["Not Purchased (0)", "Purchased (1)"], autopct='%1.1f%%', 
            colors=['#ff9999','#66b3ff'], explode=(0.05, 0), shadow=True, startangle=140)
    plt.title("Overall Purchase Distribution", fontsize=14, fontweight='bold')
    # Comment: Quick visual of the percentage of customers who converted into purchasing.

    # c) Campaign Cost vs Revenue (Scatter Plot)
    # Purpose: Analyze ROI relationship
    plt.figure(figsize=(8, 6))
    purchased_only = df_original[df_original['Purchased'] == 1]
    sns.scatterplot(data=purchased_only, x='Campaign_Cost', y='Revenue', hue='Campaign_Type', palette='deep', s=100, alpha=0.7)
    plt.title("Campaign Cost vs. Collected Revenue (Purchased = 1)", fontsize=14, fontweight='bold')
    plt.xlabel("Campaign Cost ($)", fontsize=12)
    plt.ylabel("Revenue ($)", fontsize=12)
    plt.legend(title="Campaign Type")
    # Comment: Visualizing the correlation between campaign cost investments and revenue generated for successful conversions.

    print("\n--- 9. Intelligent Business Insights ---")
    
    # Identify top influencing features
    top_feature1 = feature_df.iloc[0]['Feature']
    top_feature2 = feature_df.iloc[1]['Feature']
    
    # Identify best campaign type regarding ROI
    roi_df = df_original.groupby('Campaign_Type').agg(Cost=('Campaign_Cost', 'sum'), Revenue=('Revenue', 'sum')).reset_index()
    roi_df['ROI'] = (roi_df['Revenue'] - roi_df['Cost']) / roi_df['Cost']
    best_campaign = roi_df.loc[roi_df['ROI'].idxmax()]['Campaign_Type']
    
    # Determine if cost strongly correlates to revenue simply looking at pearson correlation
    corr = purchased_only['Campaign_Cost'].corr(purchased_only['Revenue'])
    cost_impact = "shows a strong correlation" if abs(corr) > 0.5 else "does not show a strong correlation"

    print(f"💡 Insight 1: '{top_feature1}' and '{top_feature2}' are the most influential factors for purchase decisions.")
    print(f"💡 Insight 2: Campaign cost {cost_impact} with the collected revenue for successful conversions (correlation: {corr:.2f}).")
    print(f"💡 Insight 3: '{best_campaign}' is the most beneficial campaign type overall when factoring in ROI.\n")

    # Display all charts sequentially
    print("Displaying charts... (Close the chart windows to exit)")
    plt.show()

if __name__ == "__main__":
    main()
