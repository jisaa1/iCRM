import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path =r"C:\Users\DELL\OneDrive - Ulster University\Desktop\MSC COMPUTER SCIENCE\MSC PROJECT\CRM\CRM-Enhancement\data\cleaned_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

#1. Barplot for Churn Reasons
# Calculate the percentage of customers for each churn reason
churn_counts = df["Churn Reason"].value_counts(normalize=True) * 100
churn_df = churn_counts.reset_index()
churn_df.columns = ["Churn Reason", "Percentage"]

# Create a compact bar plot
plt.figure(figsize=(6, 4))
plt.barh(churn_df["Churn Reason"], churn_df["Percentage"], color='steelblue')
plt.title('% Total Customers by Churn Reasons', fontsize=10)
plt.xlabel('% Number of Customers', fontsize=8)
plt.ylabel("Churn Reason", fontsize=8)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('CRM-Enhancement\mvisualizations\churn_reasons.png', dpi=300)
plt.show()

#2. PIE-CHART for Churn Category 
# Remove rows with empty or NaN values in the 'churn_reason' column
filtered_df = df[df["Churn Reason"].notna() & (df["Churn Reason"] != '')].copy()

# Function to categorize churn reasons as pie chart
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

# Apply the function to create the churn_category column in the new DataFrame
filtered_df['churn_category'] = filtered_df["Churn Reason"].apply(categorize_churn_reason)

# Calculate the percentage of customers for each churn category
churn_category_counts = filtered_df['churn_category'].value_counts(normalize=True) * 100

# Define professional colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(churn_category_counts, labels=churn_category_counts.index, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('% Count of Churn Label by Churn Category', fontsize=12)

# Add a legend
plt.legend(title='Churn Category', loc='upper right', bbox_to_anchor=(1.3, 1))

# Save the figure to the specified path
plt.tight_layout()
plt.savefig("CRM-Enhancement\mvisualizations\churn_category_pie.png", dpi=300)
plt.show()

#3. Count of Customers using Donut chart

# Count the number of customers for each contract type
contract_type_counts = df['Contract'].value_counts()

# Define professional colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create the donut chart
plt.figure(figsize=(4, 4))  # Smaller figure size for mobile tile
plt.pie(contract_type_counts, labels=contract_type_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(width=0.3))
plt.title('Count of Customers by Contract Type', fontsize=10)

# Save the figure to the specified path
plt.tight_layout()
plt.savefig("CRM-Enhancement\mvisualizations\contract_type_donut.png", dpi=300)
plt.show()

#4: Contract Type vs Churn
# Group by contract type and churn label to get counts
contract_churn_counts = df.groupby(["Contract", "Churn Label"]).size().unstack()

# Define colors
colors = ['#1f77b4', '#ff7f0e']

# Create the bar chart
ax = contract_churn_counts.plot(kind='bar', stacked=False, figsize=(6, 4), color=colors)

# Add title and labels
plt.title('Contract Type vs. Churn', fontsize=12)
plt.xlabel('Contract Type', fontsize=10)
plt.ylabel('Count', fontsize=10)

# Add legend
plt.legend(title='Churn Label', labels=['No', 'Yes'], loc='upper right')

# Save the figure to the specified path
plt.tight_layout()
plt.savefig('CRM-Enhancement\mvisualizations\contract_type_churn_bar.png', dpi=300)
plt.show()

# Plot: Contract Type vs. Churn to display the relationship between different contract types and churn rates
plt.figure(figsize=(6, 4))  # Smaller figure size for mobile tiles
sns.countplot(x='Contract', hue="Churn Label", data=df, palette=['#2a9d8f', '#e76f51'])
plt.title('Contract Type vs. Churn', fontsize=12)
plt.xlabel('Contract Type', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.legend(title='Churn Label', loc='upper right', fontsize=8, title_fontsize='10')
plt.tight_layout()
plt.savefig("CRM-Enhancement\mvisualizations\contract_type_churn_bar.png", dpi=300, bbox_inches='tight')
plt.show()