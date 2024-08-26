import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "CRM-Enhancement/data/cleaned_data.csv"
df = pd.read_csv(file_path)

# 1. Churn Distribution - Bar Chart
#View the proportion of customers who churned vs. those who did not, to get a clear picture of the churn rate within the dataset.
plt.figure(figsize=(8, 6))
sns.countplot(x="Churn Label", data=df, color='#2a9d8f')
plt.title('Churn Distribution', fontsize=16)
plt.xlabel('Churn', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/churn_distribution_bar.png", dpi=300, bbox_inches='tight')
plt.show()

#2. Customer Demographics, display the distrihution of customers based on age, tenure, and gender
# Function to add percentage annotations
def add_percentage(ax, total_count):
    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue
        percentage = height / total_count * 100
        ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.1f}%', 
                ha='center', va='bottom', fontsize=12, color='black')
        
# Senior Citizen Distribution with Churn Rate
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Senior Citizen', hue='Churn Label', data=df, palette=['#2a9d8f', '#e76f51'])
plt.title('Senior Citizen Distribution with Churn Rate', fontsize=16)
plt.xlabel('Senior Citizen', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Churn Label', loc='upper right', fontsize=12, title_fontsize='13')
total_count = len(df)
add_percentage(ax, total_count)
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/senior_citizen_distribution_with_churn.png", dpi=300, bbox_inches='tight')
plt.show()

# Gender Distribution with Churn Rate
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Gender', hue='Churn Label', data=df, palette=['#2a9d8f', '#e76f51'])
plt.title('Gender Distribution with Churn Rate', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Churn Label', loc='upper right', fontsize=12, title_fontsize='13')
add_percentage(ax, total_count)
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/gender_distribution_with_churn.png", dpi=300, bbox_inches='tight')
plt.show()

#Tenure distribution with Churn rate
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Tenure Months', hue='Churn Label', multiple='stack', palette=['#2a9d8f', '#e76f51'], bins=20)
plt.title('Tenure Distribution with Churn Rate', fontsize=16)
plt.xlabel('Tenure (months)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Churn Label', loc='upper right', fontsize=12, title_fontsize='13')
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/tenure_distribution_with_churn_stacked.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot: Monthly Charges vs. Churn
#Compare the distribution of monthly charges for customers who churned vs. those who did not
plt.figure(figsize=(10, 6))
sns.violinplot(x='Churn Label', y='Monthly Charges', data=df, palette=['#2a9d8f', '#e76f51'])
plt.title('Monthly Charges vs. Churn', fontsize=16)
plt.xlabel('Churn Label', fontsize=14)
plt.ylabel('Monthly Charges', fontsize=14)
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/monthly_charges_vs_churn.png", dpi=300, bbox_inches='tight')
plt.show()

#Plot: Service Usage, show the usage of various services among churned and non-churned customers, provide the code
# Select relevant service columns
services = ["Phone Service", "Internet Service", "Streaming TV", "Streaming Movies"]

# Plot 5: Service Usage vs. Churn
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Create a stacked bar chart for each service
for i, service in enumerate(services):
    ax = axs[i // 2, i % 2]
    churn_counts = df.groupby([service, 'Churn Label']).size().unstack(fill_value=0)
    churn_counts.plot(kind='bar', stacked=True, ax=ax, color=['#2a9d8f', '#e76f51'], legend=(i == 0))

    ax.set_title(f'{service} Usage vs. Churn', fontsize=14)
    ax.set_xlabel(service, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    if i == 0:
        ax.legend(title='Churn Label', loc='upper right', fontsize=12, title_fontsize='13')
    else:
        ax.legend().remove()

plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/service_usage_vs_churn1.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot: Contract Type vs. Churn, to display the relationship between different contract types and churn rates
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn Label', data=df, palette=['#2a9d8f', '#e76f51'])
plt.title('Contract Type vs. Churn', fontsize=16)
plt.xlabel('Contract Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Churn Label', loc='upper right', fontsize=12, title_fontsize='13')
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/contract_type_vs_churn.png", dpi=300, bbox_inches='tight')
plt.show()

#Plot: Payment Method vs. Churn to compare churn rates across different payment methods.
plt.figure(figsize=(10, 6))
sns.countplot(x='Payment Method', hue='Churn Label', data=df, palette=['#2a9d8f', '#e76f51'])
plt.title('Payment Method vs. Churn', fontsize=16)
plt.xlabel('Payment Method', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Churn Label', loc='upper right', fontsize=12, title_fontsize='13')
plt.tight_layout()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.savefig("CRM-Enhancement/visualization/payment_method_vs_churn1.png", dpi=300, bbox_inches='tight')
plt.show()

#plot: Heatmap of Correlations, show correlations between different features and churn.
# Convert categorical variables to numerical
# Select a sample of the dataset for quicker analysis (optional)
df_sample = df.sample(frac=0.1, random_state=42)  # Adjust the frac as needed

# Select relevant features
relevant_features = [
    "Senior Citizen", "Gender", "Tenure Months", "Monthly Charges", "Total Charges", "Churn Label", 
    "Contract", "Payment Method", "Internet Service"
]
df_relevant = df_sample[relevant_features]

# Convert categorical variables to numerical (one-hot encoding)
categorical_features = ["Senior Citizen", 'Gender', 'Contract', "Payment Method", "Internet Service", "Churn Label"]
df_encoded = pd.get_dummies(df_relevant, columns=categorical_features, drop_first=True)

# Calculate the correlation matrix
correlation_matrix = df_encoded.corr()

# Plot the heatmap with percentage annotations
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix * 100, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.5)
plt.title('Heatmap of Correlations with Percentage Annotations', fontsize=16)
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/heatmap_of_correlations.png", dpi=300, bbox_inches='tight')
plt.show()

#PLOT: SCATTERPLOT
# Select a sample of the dataset for quicker analysis (optional)
df_sample = df.sample(frac=0.1, random_state=42)  # Adjust the frac as needed

# Select relevant features
relevant_features = [
    "Senior Citizen", "Gender", "Tenure Months", "Monthly Charges", "Total Charges", "Churn Label", 
    "Contract", "Payment Method", "Internet Service"
]
df_relevant = df_sample[relevant_features]

# Convert categorical variables to numerical (one-hot encoding)
categorical_features = ["Gender", "Contract", "Payment Method", "Internet Service"]
df_encoded = pd.get_dummies(df_relevant, columns=categorical_features, drop_first=True)

# Ensure 'Churn Label' and 'Senior Citizen' are numeric
df_encoded['Churn Label'] = df_encoded['Churn Label'].map({'No': 0, 'Yes': 1})
df_encoded['Senior Citizen'] = df_encoded['Senior Citizen'].map({'No': 0, 'Yes': 1})

# Plot the pairplot
sns.pairplot(df_encoded, diag_kind='hist', hue="Churn Label", palette=['#2a9d8f', '#e76f51'], plot_kws={'alpha':0.6})
plt.suptitle('Scatterplot with Histograms and Correlations', y=1.02, fontsize=16)
plt.savefig("CRM-Enhancement/visualization/scatterplot_with_histograms_and_correlations.png", dpi=300, bbox_inches='tight')
plt.show()

#Plot: Churn Reasons, highlight the most common reasons for churn.
# Filter the dataset to include only rows where Churn Label is 'Yes'
churn_reasons = df[df["Churn Label"] == 'Yes']["Churn Reason"]

# Plot Churn Reasons
plt.figure(figsize=(12, 8))
sns.countplot(y=churn_reasons, order=churn_reasons.value_counts().index, palette='viridis')
plt.title('Most Common Reasons for Churn', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Churn Reason', fontsize=14)
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/churn_reasons.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot CLTV against churn to understand the potential revenue impact of churn and identify high-value customers at risk.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CLTV', y='Monthly Charges', hue='Churn Label', data=df, palette=['#2a9d8f', '#e76f51'], alpha=0.7)
plt.title('Customer Lifetime Value (CLTV) vs. Churn', fontsize=16)
plt.xlabel('Customer Lifetime Value (CLTV)', fontsize=14)
plt.ylabel('Monthly Charges', fontsize=14)
plt.legend(title='Churn Label', loc='upper right', fontsize=12, title_fontsize='13')
plt.tight_layout()
plt.savefig("CRM-Enhancement/visualization/cltv_vs_churn.png", dpi=300, bbox_inches='tight')
plt.show()

#Distribution of numeric variables
import matplotlib.pyplot as plt

# List of numerical columns
numerical_columns = ["Tenure Months", "Monthly Charges", "Total Charges", "Churn Score", 'CLTV']

# Color palette for the charts
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plotting distribution for each numerical column
for i, column in enumerate(numerical_columns):
    plt.figure(figsize=(8, 4))
    plt.hist(df[column], bins=30, color=colors[i], edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig("CRM-Enhancement/visualization/distribution of numeric variables.png")
    plt.show()
