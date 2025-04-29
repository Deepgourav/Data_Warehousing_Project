# Data_Warehousing_Project
# -----------------------------------------------
# TASK A: Data Cleaning and RFM Preparation
# -----------------------------------------------
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('bank_transactions.csv')  # Change the path if necessary
# -----------------------------------------------
# A1. Identify and Remove Null Values
# -----------------------------------------------
# Drop rows where any important field is null
df = df.dropna(subset=['CustomerDOB', 'CustGender', 'CustLocation', 'CustAccountBalance'])

# -----------------------------------------------
# A2. Identify and Remove Invalid Transaction Amount Values
# -----------------------------------------------
# Remove transactions where amount is zero or negative
df = df[df['TransactionAmount (INR)'] > 0]
# -----------------------------------------------
# A3. Identify and Remove Invalid Age Values
# -----------------------------------------------
# Convert CustomerDOB and TransactionDate to datetime
df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], dayfirst=True, errors='coerce')
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], dayfirst=True, errors='coerce')
# Calculate Age assuming transactions mostly occurred in 2016
reference_year = 2016
df['Age'] = reference_year - df['CustomerDOB'].dt.year
# Remove unrealistic ages (below 18 or above 100)
df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]

# -----------------------------------------------
# A4. Display the Top 5 Locations where the Maximum Number of Transactions Occurred
# -----------------------------------------------
top_locations = df['CustLocation'].value_counts().head(5)
print(top_locations)
# Optional: Plotting Top 5 Locations
top_locations.plot(kind='bar', figsize=(8,5), title='Top 5 Locations by Transactions')
plt.xlabel('Location')
plt.ylabel('Number of Transactions')
plt.show()

# -----------------------------------------------
# A5. Write a Query to Define and Calculate RFM Values per Customer
# -----------------------------------------------
# Define reference date (latest transaction date)
reference_date = df['TransactionDate'].max()
# Calculate RFM values
rfm = df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (reference_date - x.max()).days,  # Recency
    'TransactionID': 'count',                                      # Frequency
    'TransactionAmount (INR)': 'sum'                               # Monetary
}).reset_index()
# Rename the columns for clarity
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
# Display RFM Table
print(rfm.head())

# -----------------------------------------------
# A6. Check Distribution of Recency, Frequency, and Monetary Values
# -----------------------------------------------
# Plotting original distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(rfm['Recency'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Recency Distribution')
axes[0].set_xlabel('Recency (days)')
axes[0].set_ylabel('Number of Customers')
axes[1].hist(rfm['Frequency'], bins=50, color='lightgreen', edgecolor='black')
axes[1].set_title('Frequency Distribution')
axes[1].set_xlabel('Frequency')
axes[2].hist(rfm['Monetary'], bins=50, color='salmon', edgecolor='black')
axes[2].set_title('Monetary Distribution')
axes[2].set_xlabel('Monetary Value (INR)')
plt.tight_layout()
plt.show()

# -----------------------------------------------
# A7. Briefly Discuss the Issue of Skewness and Remove Skewness
# -----------------------------------------------
# Apply log transformation to fix skewness
rfm['Recency_log'] = np.log1p(rfm['Recency'])
rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
# Plotting transformed distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(rfm['Recency_log'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Log-Transformed Recency')
axes[0].set_xlabel('Log(Recency + 1)')
axes[1].hist(rfm['Frequency_log'], bins=50, color='lightgreen', edgecolor='black')
axes[1].set_title('Log-Transformed Frequency')
axes[1].set_xlabel('Log(Frequency + 1)')
axes[2].hist(rfm['Monetary_log'], bins=50, color='salmon', edgecolor='black')
axes[2].set_title('Log-Transformed Monetary')
axes[2].set_xlabel('Log(Monetary + 1)')
plt.tight_layout()
plt.show()
# -----------------------------------------------
# END of TASK A
# -----------------------------------------------
# -----------------------------------------------
# TASK B1–B4: Customer Segmentation and Data Mart Design
# -----------------------------------------------
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Assuming you have already prepared your RFM data:
# rfm dataframe must contain: CustomerID, Recency_log, Frequency_log, Monetary_log
# -----------------------------------------------
# B2: Find the Best Value of K (Number of Clusters)
# -----------------------------------------------
# Feature Selection
X = rfm[['Recency_log', 'Frequency_log', 'Monetary_log']]
# Empty lists to collect WCSS and Silhouette Scores
wcss = []
silhouette_scores = []
# Trying K values from 2 to 10
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # WCSS
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
# Plotting Elbow Plot (WCSS vs K)
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='-', color='blue')
plt.title('Figure 1: Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.xticks(K_range)
plt.tight_layout()
plt.show()
# Plotting Silhouette Score Plot (Silhouette Score vs K)
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color='green')
plt.title('Figure 2: Silhouette Score to Determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.xticks(K_range)
plt.tight_layout()
plt.show()


# -----------------------------------------------
# B3: Apply K-Means Clustering with K=4
# -----------------------------------------------
# Apply KMeans with optimal K=4 (as observed from plots)
kmeans_final = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans_final.fit_predict(X)
# Display a few rows with assigned Cluster
print(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster']].head())

# -----------------------------------------------
# B3 (continued): Profile the Clusters
# -----------------------------------------------
# Cluster Profile: Average RFM values per Cluster
cluster_profile = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()
print(cluster_profile)

# -----------------------------------------------
# B4: Data Mart Design
# -----------------------------------------------
# (Conceptual — no coding needed, but if you want to create the Data Mart structure:)
# Example: Create basic Data Mart structure
rfm['AgeGroup'] = pd.cut(rfm['Recency'], bins=[0, 25, 50, 75, 100, np.inf],
                         labels=['0-25 days', '26-50 days', '51-75 days', '76-100 days', '100+ days'])
data_mart = rfm[['CustomerID', 'Cluster', 'Recency', 'Frequency', 'Monetary', 'AgeGroup']]
print(data_mart.head())
# (In a real Data Mart, additional fields like Gender, Location would be added too)

# -----------------------------------------------
# END of TASK B
# -----------------------------------------------

