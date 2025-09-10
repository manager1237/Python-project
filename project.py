import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

df=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
df.head()
df.tail()
print("Dataset Information:")
print(df.info())
# Display of the columns of the dataset
print("\nColumns in the dataset:")
print(df.columns.tolist())
df.shape
df.dtypes
print("Summary Statistics:")
summary_stats = df.describe(include='all').T

# Show as a table for better readability
display(summary_stats) 
# duplicated rows in the dataset
df.duplicated().any()
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['year'] = df['publish_time'].dt.year
df['year'].describe()
df['abstract_length'] = df['abstract'].astype(str).apply(len)
df['abstract_length'].describe()

df['author_count'] = df['authors'].astype(str).apply(lambda x: len(x.split(';')) if x != 'nan' else 0)
df['author_count'].describe()
missing_values = df.isnull().sum().sort_values()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Missing (%)': missing_percent
})

# Bar plot for missing values percentage
missing_percent = (df.isnull().sum() / len(df)) * 100

plt.figure(figsize=(12, 6))
missing_percent[missing_percent > 0].sort_values(ascending=False).plot(kind='bar', color='salmon')
plt.ylabel("Missing Value Percentage (%)")
plt.title("Percentage of Missing Values by Column")
plt.xticks(rotation=45)
plt.show()

numerical = df.select_dtypes(include=['float64', 'int64'])
for i in numerical.columns:
    print(f"Skewness of {i}: {df[i].skew()}")
    print(f"Standart deviation of {i}: {df[i].std()}")
    # Outlier detection using IQR method
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75) 
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[i] < lower_bound) | (df[i] > upper_bound)]
    print(f"Number of outliers in {i}: {len(outliers)}")
    print("------------------------------------")

    plt.figure(figsize=(10, 4))
sns.boxplot(x=df['abstract_length'], color='lightgreen')
plt.title("Abstract Uzunluğunun Boxplot Görselleştirmesi")
plt.xlabel("abstract_length")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['author_count'], color='lightgreen')
plt.title("Visualization of Author Count with Boxplot")
plt.xlabel("Author Count")
plt.show()
# Only numerical columns
numerical_cols = df.select_dtypes(include='number')

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Bar plot for distribution of publication sources
source_counts = df['source_x'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=source_counts.index, y=source_counts.values, color='skyblue')
plt.title("Distribution of Publication Sources (source_x)")
plt.xlabel("Source")
plt.ylabel("Number of Publications")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Pie chart for distribution of publication licenses
license_counts = df['license'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(license_counts, labels=license_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Distribution of Publication Licenses")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
top_journals = df['journal'].value_counts().head(10)
#  Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(y=top_journals.index, x=top_journals.values, palette="Blues_d")
plt.title("Top 10 Journals by Number of Publications")
plt.xlabel("Number of Publications")
plt.ylabel("Journal")
plt.tight_layout()
plt.show()
