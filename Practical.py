# ---------------------- 📌 Import Libraries ----------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# ---------------------- 1️⃣ Load Dataset ----------------------
file_path = r"C:\Users\Shreyash Musmade\Desktop\Practical\MIDS\MIDS_Prac-1\titanic3.xls"

try:
    df = pd.read_excel(r"C:\Users\Shreyash Musmade\Desktop\Practical\MIDS\MIDS_Prac-1\titanic_preprocessed.xlsx")
    print("\n[INFO] Dataset Loaded Successfully!")  
except Exception as e:
    print("\n[ERROR] Failed to Load Dataset:", e)

# ---------------------- 2️⃣ Basic Information ----------------------
print("\n[INFO] Dataset Info:")
df.info()

# Dataset Shape
print(f"\n[INFO] Dataset Shape: {df.shape}")

# Display First 5 Rows
print("\n[INFO] First 5 Rows:")
print(df.head())

# ---------------------- 3️⃣ Summary Statistics ----------------------
print("\n[INFO] Summary of Numerical Features:")
print(df.describe())

print("\n[INFO] Summary of Categorical Features:")
print(df.describe(include=['O']))  # 'O' means object (categorical data)

# ---------------------- 4️⃣ Missing Values ----------------------
print("\n[INFO] Missing Values Count:")
print(df.isnull().sum())

# Visualizing missing values
plt.figure(figsize=(10, 5))
msno.bar(df)
plt.title("Missing Values Overview")
plt.show()

# ---------------------- 5️⃣ Check for Duplicates ----------------------
print("\n[INFO] Duplicate Rows:", df.duplicated().sum())

# ---------------------- 6️⃣ Data Distribution ----------------------
# Histogram for numerical features
df.hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Countplot for categorical columns (e.g., Gender)
plt.figure(figsize=(5, 3))
sns.countplot(x=df['sex'])
plt.title("Count of Passengers by Gender")
plt.show()

# ---------------------- 7️⃣ Correlation Analysis ----------------------
# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Heatmap visualization
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# ---------------------- 8️⃣ Outlier Detection ----------------------
# Boxplot for 'fare'
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['fare'])
plt.title("Boxplot of Fare Prices")
plt.show()

print("\n[INFO] Data Exploration Completed Successfully!")  



# ---------------------- 3️⃣ Handling Missing Values ----------------------

# 🔹 3.1 Drop Columns with Too Many Missing Values
threshold = 0.6  # If more than 60% values are missing, drop the column
missing_ratio = df.isnull().sum() / len(df)

columns_to_drop = missing_ratio[missing_ratio > threshold].index
df.drop(columns=columns_to_drop, inplace=True)
print(f"\n[INFO] Dropped Columns: {list(columns_to_drop)}")

# 🔹 3.2 Fill Missing Values for Numerical Columns
num_cols = df.select_dtypes(include=['number']).columns

for col in num_cols:
    if df[col].isnull().sum() > 0:  
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"[INFO] Filled missing values in '{col}' with median ({median_value})")

# 🔹 3.3 Fill Missing Values for Categorical Columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    if df[col].isnull().sum() > 0:  
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"[INFO] Filled missing values in '{col}' with mode ('{mode_value}')")

# 🔹 3.4 Verify That No Missing Values Remain
print("\n[INFO] Missing Values After Handling:")
print(df.isnull().sum())







# ---------------------- 4️⃣ Handling Categorical Data ----------------------

# 🔹 4.1 Identify Categorical Columns
cat_cols = df.select_dtypes(include=['object']).columns
print("\n[INFO] Categorical Columns:", list(cat_cols))

# 🔹 4.2 Label Encoding for Binary Categorical Features
from sklearn.preprocessing import LabelEncoder

binary_cols = ['sex']  # Columns with only two categories
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    print(f"[INFO] Applied Label Encoding to '{col}'")

# 🔹 4.3 One-Hot Encoding for Multi-Class Categorical Features
multi_class_cols = ['embarked']  # Modify based on dataset
df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)
print(f"[INFO] Applied One-Hot Encoding to: {multi_class_cols}")

# 🔹 4.4 Ordinal Encoding for Ordered Categorical Features
ordinal_cols = {'pclass': {1: 1, 2: 2, 3: 3}}  # Modify mapping if needed

for col, mapping in ordinal_cols.items():
    df[col] = df[col].map(mapping)
    print(f"[INFO] Applied Ordinal Encoding to '{col}'")

# 🔹 4.5 Verify Categorical Data Encoding
print("\n[INFO] Categorical Data After Encoding:")
print(df.head())

# ---------------------- 5️⃣ Feature Scaling ----------------------

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 🔹 5.1 Selecting Numerical Columns for Scaling
num_features = ['age', 'fare']  # Modify based on dataset availability
df_selected = df[num_features]

# 🔹 5.2 Apply Min-Max Scaling (Normalization)
minmax_scaler = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(df_selected), columns=df_selected.columns)

print("\n[INFO] Min-Max Scaled Data (0 to 1 Range):")
print(df_minmax_scaled.head())

# 🔹 5.3 Apply Standardization (Z-score Normalization)
standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(standard_scaler.fit_transform(df_selected), columns=df_selected.columns)

print("\n[INFO] Standardized Data (Mean = 0, Std Dev = 1):")
print(df_standardized.head())

# 🔹 5.4 Replace Scaled Values in Original Dataset
df[num_features] = df_minmax_scaled  # Choose either min-max or standardized scaling
print("\n[INFO] Feature Scaling Applied Successfully!")


# ---------------------- 6️⃣ Outlier Detection & Removal ----------------------

# 🔹 6.1 Detect Outliers Using Boxplots
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['fare'])
plt.title("Boxplot of Fare Prices (Before Outlier Removal)")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age (Before Outlier Removal)")
plt.show()

# 🔹 6.2 Remove Outliers Using IQR Method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

# Apply to 'fare' and 'age' columns
df = remove_outliers_iqr(df, 'fare')
df = remove_outliers_iqr(df, 'age')

# 🔹 6.3 Verify Outlier Removal Using Boxplots
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['fare'])
plt.title("Boxplot of Fare Prices (After Outlier Removal)")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age (After Outlier Removal)")
plt.show()

print("\n[INFO] Outlier Detection & Removal Completed!")


# ---------------------- 7️⃣ Save the Preprocessed Dataset ----------------------

# Define the file path where the dataset will be saved
save_path_csv = r"C:\Users\Shreyash Musmade\Desktop\Practical\archive\titanic_preprocessed.csv"
save_path_excel = r"C:\Users\Shreyash Musmade\Desktop\Practical\archive\titanic_preprocessed.xlsx"

try:
    # 🔹 7.1 Save as CSV File
    df.to_csv(save_path_csv, index=False)
    print(f"\n[INFO] Dataset successfully saved as CSV at: {save_path_csv}")

    # 🔹 7.2 Save as Excel File (Optional)
    df.to_excel(save_path_excel, index=False)
    print(f"[INFO] Dataset successfully saved as Excel at: {save_path_excel}")

except Exception as e:
    print("\n[ERROR] Failed to save dataset:", e)

# 🔹 7.3 Verify the Saved File Exists
import os
if os.path.exists(save_path_csv):
    print("\n[INFO] CSV file saved successfully and verified!")
else:
    print("\n[ERROR] CSV file not found. Check the save path.")

if os.path.exists(save_path_excel):
    print("[INFO] Excel file saved successfully and verified!")
else:
    print("[ERROR] Excel file not found. Check the save path.")
