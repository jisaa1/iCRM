import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file
file_path = "CRM-Enhancement/data/Telco_customer_churn.xlsx"
df = pd.read_excel(file_path)

# View the structure of the data
print("Data Structure:")
df.info()

# Check data types
print(df.dtypes)

# Check distribution of target variable
target_variable = "Churn Label"  
print("\nDistribution of Target Variable:")
print(df[target_variable].value_counts())

#Visualization of the target variable
sns.countplot(x=df[target_variable])
plt.title("Distribution of Target Variable")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.savefig("Churndistribution.png")
plt.show()

# Check for duplicate records
print(df.duplicated().sum()) # There are no duplicate values in the dataset

# Print unique values in categorical variables
print("\nUnique values in categorical variables:")
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    unique_values = df[col].unique()
    print(f"{col}: {unique_values}") #There are only valid records in the columns 

#Checking the discrepancies in datatypes
#check the datatype of column 'Total Charges'
dt_tc = df["Total Charges"].dtype
print(dt_tc) #Total Charges, column is object, need to change that to numeric

#Change Total Charges to numeric
# Convert 'Total Charges' to a numeric type
if "Total Charges" in df.columns:
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors='coerce')

#Check Total Charges is changed to numeric
dt_tc = df["Total Charges"].dtype
print(dt_tc)

#Removing the unnecessary values in the dataset
df.drop(columns=["Zip Code", "Latitude", "Longitude", "Churn Value"], inplace=True)
df.drop(columns=["Count"], inplace=True)
df.drop(columns=["Lat Long"], inplace=True)
#Check for outliers using boxplot (select numerical columns)
#First display only numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Columns:")
print(numerical_columns)

#Visualise the boxplot for numerical variables
#Select numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# Create a figure with subplots
num_cols = len(numerical_columns)
fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, num_cols * 3))
fig.suptitle("Boxplots of Numerical Columns", fontsize=16)

# Create subplots for each numerical column
for i, col in enumerate(numerical_columns):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(f"Boxplot of {col}")

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("boxplots_numerical_columns.png")
plt.show()

# Check for missing values in the dataset
print("\nMissing Values:")
print(df.isna().sum())

# Replace empty strings and spaces with NaN
df.replace(["", " "], np.nan, inplace=True)

# Check again for missing values
print("\nMissing Values after replacing empty strings with NaN:")
print(df.isna().sum())  # TotalCharges and Churn Reason have missing values

# Handle missing values, impute missing values using mean for Total Charges, becuase Churn Reason is valid
df["Total Charges"].fillna(df["Total Charges"].mean(), inplace=True)

# Check again for missing values
print("\nMissing Values after replacing NaN with mean:")
print(df.isna().sum())  # missing values are filled with mean for TotalCharges

# Finally save the cleaned data to a new file
cleaned_file_path = "CRM-Enhancement/data/cleaned_data.csv"
df.to_csv(cleaned_file_path, index=False)
print("\nCleaned data saved")