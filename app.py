from flask import Flask, jsonify
import pandas as pd
import os
from io import StringIO



app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask! Jisa"

# Load the dataset
#file_path = ("C:/Working folder- Jisa/data/cleaned_data.csv")
df = pd.read_csv('https://dpapifiles.blob.core.windows.net/apifiles/cleaned_data.csv')

# Summary of Churned Customers
@app.route('/summary', methods=['GET'])
def get_summary():
    total_customers = len(df)
    churned_customers = df[df["Churn Label"] == 'Yes'].shape[0]
    churn_rate = (churned_customers / total_customers) * 100
    churn_customers = churn_data[(churn_data['ChurnStatus'] == 'Churn') & (churn_data['ChurnProbability'] > 80)]
    
    

    filtered_df = df[df["Churn Reason"].notna() & (df["Churn Reason"] != '')].copy()

    def categorize_churn_reason(reason):
        reason = reason.lower()
        if 'competitor' in reason:
            return 'Competitor'
        elif 'price' in reason or 'charge' in reason:
            return 'Price'
        elif 'attitude' in reason or 'service' in reason or 'support' in reason:
            return 'Service'
        elif 'product' in reason:
            return 'Product'
        elif 'network' in reason or 'reliability' in reason:
            return 'Network'
        elif 'moved' in reason:
            return 'Relocation'
        elif 'poor expertise' in reason:
            return 'Support'
        elif 'deceased' in reason:
            return 'Other'
        else:
            return 'Other'

    filtered_df['churn_category'] = filtered_df["Churn Reason"].apply(categorize_churn_reason)
    churn_category_counts = filtered_df['churn_category'].value_counts(normalize=True) * 100
    churn_category_counts = churn_category_counts.round(2)

    churn_reasons = df[df["Churn Label"] == 'Yes']["Churn Reason"]
    churn_reason_counts = churn_reasons.value_counts()

    contract_type_counts = df['Contract'].value_counts() 

    summary = {
        'Churn_Rate': f'{churn_rate:.2f}%',
        'Total_Customers': total_customers,
        'Churned_Customers': churned_customers,
        'Churn_Probability': churn_customers.shape[0],
        'Churn_Category' : churn_category_counts.to_dict(),
        'Churn_reasons' : churn_reason_counts.to_dict(),
        'Churn_contracts' : contract_type_counts.to_dict()
    }
    return jsonify(summary)

# Churned Category - Pie Chart
@app.route('/data/churn_category', methods=['GET'])
def get_churn_category_data():
    filtered_df = df[df["Churn Reason"].notna() & (df["Churn Reason"] != '')].copy()

    def categorize_churn_reason(reason):
        reason = reason.lower()
        if 'competitor' in reason:
            return 'Competitor'
        elif 'price' in reason or 'charge' in reason:
            return 'Price'
        elif 'attitude' in reason or 'service' in reason or 'support' in reason:
            return 'Service'
        elif 'product' in reason:
            return 'Product'
        elif 'network' in reason or 'reliability' in reason:
            return 'Network'
        elif 'moved' in reason:
            return 'Relocation'
        elif 'poor expertise' in reason:
            return 'Support'
        elif 'deceased' in reason:
            return 'Other'
        else:
            return 'Other'

    filtered_df['churn_category'] = filtered_df["Churn Reason"].apply(categorize_churn_reason)
    churn_category_counts = filtered_df['churn_category'].value_counts(normalize=True) * 100
    churn_category_counts = churn_category_counts.round(2)

    return jsonify(churn_category_counts.to_dict())

# Most Common Reasons for Churn
@app.route('/data/churn_reasons', methods=['GET'])
def get_churn_reasons_data():
    churn_reasons = df[df["Churn Label"] == 'Yes']["Churn Reason"]
    churn_reason_counts = churn_reasons.value_counts()
    return jsonify(churn_reason_counts.to_dict())

# Contract Type vs Churn
@app.route('/data/contract_type_counts', methods=['GET'])
def get_contract_type_counts():
    contract_type_counts = df['Contract'].value_counts()
    return jsonify(contract_type_counts.to_dict())

# Load the churn data from the JSON file
#file_path1 = pd.read_json("C:/Working folder- Jisa/data/churn_probabilities_all_customers.json", lines=True)
churn_data = pd.read_json("https://dpapifiles.blob.core.windows.net/apifiles/churn_probabilities_all_customers.json", lines=True)
# Endpoint to get all customers with Churn status
@app.route('/churn', methods=['GET'])
def get_churn_data():
    # Filter the data to include only customers with Churn status
    # churn_customers = churn_data[churn_data['ChurnStatus'] == 'Churn']
    # result = churn_customers.to_dict(orient='records')

    # Define the bins and corresponding labels
    bins = [70, 75, 80, 85, 90, 95, 100]
    labels = ['70-75', '76-80', '81-85', '86-90', '91-95', '96-100']
    # Create a new column for the binned ranges
    churn_data['ChurnProbabilityRange'] = pd.cut(churn_data['ChurnProbability'], bins=bins, labels=labels, right=False)

    # Group by ChurnProbability and count the number of customers in each group
    grouped_counts = churn_data.groupby('ChurnProbabilityRange')['CustomerID'].count().reset_index()

    # Rename the columns for clarity
    grouped_counts.columns = ['ChurnProbabilityRange', 'CustomerCount']
    result = grouped_counts.to_dict(orient='records')
    return jsonify(result)

@app.route('/churnprobability', methods=['GET'])
def get_churn_probability():
    # Filter the data to include only customers with Churn status
    churn_customers = churn_data[(churn_data['ChurnStatus'] == 'Churn') & (churn_data['ChurnProbability'] > 80)]
    
    return  jsonify({"count": churn_customers.shape[0]})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5002)

