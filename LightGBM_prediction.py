#C:/Working Folder/CRM-MLModel/data/cleaned_data.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data_path = "C:/Working Folder/CRM-MLModel/data/cleaned_data.csv"
data = pd.read_csv(data_path)

# Feature engineering: Add interaction term and handle division by zero
data["TotalChargesPerMonth"] = np.where(data["Tenure Months"] != 0, data["Total Charges"] / data["Tenure Months"], 0)

# Drop columns not needed for modeling
columns_to_drop = ['CustomerID', 'Country', 'State', 'City', "Churn Reason"]
data_cleaned = data.drop(columns=columns_to_drop)
data_cleaned.reset_index(drop=True, inplace=True)

# Handle missing values by filling with a placeholder
data_cleaned = data_cleaned.fillna('None')

# Encode categorical variables
label_encoders = {}
for column in data_cleaned.select_dtypes(include=['object']).columns:
    if column != "Churn Label":
        le = LabelEncoder()
        data_cleaned[column] = le.fit_transform(data_cleaned[column])
        label_encoders[column] = le

# Encode the target variable 'Churn Label'
target_le = LabelEncoder()
data_cleaned["Churn Label"] = target_le.fit_transform(data_cleaned["Churn Label"])

# Ensure indices are aligned after encoding
data_cleaned.reset_index(drop=True, inplace=True)

# Define features and target
X = data_cleaned.drop(columns=["Churn Label"])
y = data_cleaned["Churn Label"]

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Ensure indices are aligned after resampling
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled = pd.Series(y_resampled)
X_resampled.reset_index(drop=True, inplace=True)
y_resampled.reset_index(drop=True, inplace=True)

# Feature scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Convert back to DataFrame to maintain indices
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Ensure indices are aligned after splitting
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Initialize the LightGBM model
lgbm = LGBMClassifier(random_state=42)

# Perform 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(lgbm, X_resampled, y_resampled, cv=kfold, scoring='roc_auc')
print(f'Cross-Validation ROC AUC Scores: {cv_results}')
print(f'Mean CV ROC AUC Score: {cv_results.mean():.4f}')

# Save cross-validation results
output_dir = "C:/Working Folder/CRM-MLModel/output"
os.makedirs(output_dir, exist_ok=True)

cv_results_df = pd.DataFrame(cv_results, columns=['ROC AUC'])
cv_results_path = os.path.join(output_dir, "cv_results_lightgbm.csv")
cv_results_df.to_csv(cv_results_path, index=False)
print(f"Cross-validation results saved to {cv_results_path}")

# Train the LightGBM model on the full training data
lgbm.fit(X_train, y_train)

# Predict probabilities of churn
y_pred_prob = lgbm.predict_proba(X_test)[:, 1]

# Predict binary labels
y_pred = lgbm.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model ROC AUC: {roc_auc:.4f}")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Precision: {precision:.4f}")
print(f"Model Recall: {recall:.4f}")
print(f"Model F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred))

# Save evaluation metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Score': [accuracy, precision, recall, f1, roc_auc]
})
metrics_path = os.path.join(output_dir, "metrics_lightgbm.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"Evaluation metrics saved to {metrics_path}")

# Plot and save the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LightGBM')
plt.legend(loc="lower right")

roc_curve_path = os.path.join(output_dir, "roc_curve_lightgbm.png")
plt.savefig(roc_curve_path)
plt.close()

print(f"Saved ROC curve for LightGBM to {roc_curve_path}")

# Save the model as a joblib file
model_path = os.path.join(output_dir, "lightgbm_model.joblib")
joblib.dump(lgbm, model_path)
print(f"Model saved to {model_path}")

# Ensure the indices are aligned correctly with the original data
# We map back to the original dataset using the indices from y_test
original_indices = y_test.index
customer_ids = data.loc[original_indices, 'CustomerID']

# Create a DataFrame to hold customer details and their churn probabilities
probabilities_df = pd.DataFrame({
    'CustomerID': customer_ids,
    'ChurnProbability': (y_pred_prob * 100).round(2),  # Probability with 2 decimals
    'ChurnStatus': ['Churn' if prob > 0.5 else 'No Churn' for prob in y_pred_prob]
})

# Save the churn probabilities and status for all customers as a JSON file
json_path = os.path.join(output_dir, "churn_probabilities_all_customers.json")
probabilities_df.to_json(json_path, orient='records', lines=True)
print(f"Churn probabilities and status for all customers saved to {json_path}")

# Plot and save the performance metrics as an image
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
tbl = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
plt.savefig(os.path.join(output_dir, "metrics_lightgbm.png"))
plt.close()

print(f"Performance metrics saved as an image to {os.path.join(output_dir, 'metrics_lightgbm.png')}")
