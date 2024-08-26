import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load the dataset
file_path = "C:/Users/DELL/OneDrive - Ulster University/Desktop/MSC COMPUTER SCIENCE/MSC PROJECT/CRM/CRM-Enhancement/data/cleaned_data_new.csv"
data = pd.read_csv(file_path)

# Feature engineering
data["TotalChargesPerMonth"] = np.where(data["Tenure Months"] != 0, data["Total Charges"] / data["Tenure Months"], 0)

# Prepare the data
features_to_include = ['Country', 'State', 'Gender', 'Senior Citizen', 'Partner', 
                       'Dependents', 'Tenure Months', 'Phone Service', 'Multiple Lines', 'Internet Service',
                       'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
                       'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method', 'Monthly Charges', 
                       'Total Charges', 'CLTV', 'TotalChargesPerMonth']

data = data[features_to_include + ['Churn Label']]

data_copy = data[features_to_include + ['Churn Label']]

# Encode categorical variables
label_encoders_copy = {}
for column in data_copy.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data_copy[column] = le.fit_transform(data_copy[column])
    label_encoders_copy[column] = le

# Separate features and target
X_copy = data_copy.drop(columns=["Churn Label"])
y_copy = data_copy["Churn Label"]

# Train a Random Forest model for feature importance
rf_copy = RandomForestClassifier(n_estimators=100, random_state=42)
rf_copy.fit(X_copy, y_copy)

# Get feature importances
feature_importances_copy = rf_copy.feature_importances_

# Create a DataFrame for feature importances
importance_df_copy = pd.DataFrame({
    'Feature': X_copy.columns,
    'Importance': feature_importances_copy
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_copy.head(10))
plt.title('Top 10 Feature Importance Ranking for Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Save the plot as an image
plt.savefig("C:/Users/DELL/OneDrive - Ulster University/Desktop/MSC COMPUTER SCIENCE/MSC PROJECT/CRM/CRM-Enhancement/allmodeloutput/feature_importance1.png")

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
X = data.drop(columns=["Churn Label"])
y = data["Churn Label"]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Create directories for saving models 
models_dir = "C:/Users/DELL/OneDrive - Ulster University/Desktop/MSC COMPUTER SCIENCE/MSC PROJECT/CRM/CRM-Enhancement/models"
visualization_dir = "C:/Users/DELL/OneDrive - Ulster University/Desktop/MSC COMPUTER SCIENCE/MSC PROJECT/CRM/CRM-Enhancement/allmodeloutput"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(visualization_dir, exist_ok=True)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize results dictionary
cv_results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

# Function to save confusion matrix as a table and as an image
def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    
    # Save as CSV
    confusion_matrix_csv_path = os.path.join(visualization_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.csv")
    cm_df.to_csv(confusion_matrix_csv_path)
    print(f"Saved confusion matrix CSV for {model_name} to {confusion_matrix_csv_path}")
    
    # Save as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    confusion_matrix_image_path = os.path.join(visualization_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.savefig(confusion_matrix_image_path)
    plt.close()
    print(f"Saved confusion matrix image for {model_name} to {confusion_matrix_image_path}")

# Train, evaluate, and save each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Perform cross-validation
    scores = cross_validate(model, X_resampled, y_resampled, cv=cv, 
                            scoring=['accuracy', 'precision', 'recall', 'f1'],
                            return_train_score=False)
    
    # Calculate mean and std for each metric
    accuracy_mean = scores['test_accuracy'].mean()
    accuracy_std = scores['test_accuracy'].std()
    
    precision_mean = scores['test_precision'].mean()
    precision_std = scores['test_precision'].std()
    
    recall_mean = scores['test_recall'].mean()
    recall_std = scores['test_recall'].std()
    
    f1_mean = scores['test_f1'].mean()
    f1_std = scores['test_f1'].std()
    
    # Append results
    cv_results['Model'].append(model_name)
    cv_results['Accuracy'].append(f"{accuracy_mean:.3f}±{accuracy_std:.3f}")
    cv_results['Precision'].append(f"{precision_mean:.3f}±{precision_std:.3f}")
    cv_results['Recall'].append(f"{recall_mean:.3f}±{recall_std:.3f}")
    cv_results['F1-score'].append(f"{f1_mean:.3f}±{f1_std:.3f}")
    
    # Save the model
    model_path = os.path.join(models_dir, f"{model_name.replace(' ', '_').lower()}_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {model_name} to {model_path}")
    
    # Predict probabilities and labels
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Save confusion matrix as both CSV and image
    save_confusion_matrix(y_test, y_pred, model_name)

# Convert results to DataFrame
cv_results_df = pd.DataFrame(cv_results)

# Save cross-validation results
results_path = os.path.join(visualization_dir, "classification_metrics_results.csv")
cv_results_df.to_csv(results_path, index=False)
print(f"Classification metrics saved to {results_path}")

# Save classification metrics as a table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=cv_results_df.values, colLabels=cv_results_df.columns, cellLoc='center', loc='center')
table_image_path = os.path.join(visualization_dir, "classification_metrics_table.png")
plt.savefig(table_image_path)
plt.show()
print(f"Classification metrics table saved to {table_image_path}")

# Plot and save ROC curves
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Different Models')
plt.legend(loc="lower right")

# Save the combined ROC curve plot
roc_curve_path = os.path.join(visualization_dir, "roc_curve_all_models.png")
plt.savefig(roc_curve_path)
plt.show()
print(f"Saved combined ROC curve to {roc_curve_path}")
