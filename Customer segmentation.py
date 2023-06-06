import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import KElbowVisualizer
import warnings
import sys

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Loading Data
data = pd.read_csv('market.csv')
data.head()

# Data Cleaning
data.info()

# Handle missing values by dropping rows with missing values
data = data.dropna()

# Convert 'Dt_Customer' column to datetime format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])

# Calculate 'Customer_For' as the number of days since the customer joined
dates = []
for i in data['Dt_Customer']:
    i = i.date()
    dates.append(i)

days = []
d1 = max(dates)
for i in dates:
    delta = d1 - i
    days.append(delta)

data['Customer_For'] = days
data['Customer_For'] = pd.to_numeric(data['Customer_For'], errors='coerce')

# Explore categorical features: 'Marital_Status' and 'Education'
data['Marital_Status'].value_counts()
data['Education'].value_counts()

# Calculate 'Age' as the difference between the current year (2021) and 'Year_Birth'
data['Age'] = 2021 - data['Year_Birth']

# Calculate total amount spent by summing up different spending categories
data['Spent'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

# Create a new feature 'Living_With' to categorize marital status
data['Living_With'] = data['Marital_Status'].replace({
    'Married': 'Partner',
    'Together': 'Partner',
    'Absurd': 'Alone',
    'Widow': 'Alone',
    'YOLO': 'Alone',
    'Divorced': 'Alone',
    'Single': 'Alone'
})

# Calculate 'Children' as the sum of 'Kidhome' and 'Teenhome'
data['Children'] = data['Kidhome'] + data['Teenhome']

# Calculate 'Family_Size' by considering the 'Living_With' category and number of children
data['Family_Size'] = data['Living_With'].replace({
    'Alone': 1,
    'Partner': 2
}) + data['Children']

# Create a binary feature 'Is_Parent' indicating whether the customer has children or not
data['Is_Parent'] = np.where(data.Children > 0, 1, 0)

# Map 'Education' categories to more meaningful names
data['Education'] = data['Education'].replace({
    'Basic': 'Undergraduate',
    '2n Cycle': 'Undergraduate',
    'Graduation': 'Graduate',
    'Master': 'PostGraduate',
    'PhD': 'PostGraduate'
})

# Rename columns for better clarity
data = data.rename(columns={'MntWines': 'Wines', 'MntFruits': 'Fruits', 'MntMeatProducts': 'Meat',
                            'MntFishProducts': 'Fish', 'MntSweetProducts': 'Sweets', 'MntGoldProds': 'Gold'})

# Drop unnecessary columns
to_drop = ['Dt_Customer', 'Marital_Status', 'Z_CostContact', 'Z_Revenue', 'ID', 'Year_Birth']
data = data.drop(to_drop, axis=1)

# Explore statistical summary of the data
data.describe()

# Set seaborn settings
sns.set(rc={'axes.facecolor': '#FFF9ED', 'figure.facecolor': '#FFF9ED'})
pallet = ['#682F2F', '#9E726F', '#D6B2B1', '#B9C0C9', '#9F8A78', '#F3AB60']

# Define the features to plot
To_Plot = ['Income', 'Recency', 'Customer_For', 'Age', 'Spent', 'Is_Parent']

# Create pair plots to visualize relationships between the selected features
plt.figure()
sns.pairplot(data[To_Plot], hue='Is_Parent', palette=(['#682F2F', '#F3AB60']))
plt.show()

# Data Preprocessing

# Identify categorical columns
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

# Perform label encoding on the categorical columns
LE = LabelEncoder()
for i in object_cols:
    data[i] = data[[i]].apply(LE.fit_transform)

# Create a copy of the preprocessed data
ds = data.copy()

# Drop additional columns for clustering
cols_del = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)

# Dimensionality Reduction using PCA

# Perform PCA with 3 components
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(['col1', 'col2', 'col3']))

# Explore the transformed data
PCA_ds.describe().T

# Visualize the data in a 3D plot
x = PCA_ds['col1']
y = PCA_ds['col2']
z = PCA_ds['col3']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='green', marker='o')
ax.set_title('A 3D projection of data in reduced dimension')
plt.show()

# Clustering

# Determine the optimal number of clusters using the elbow method
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()

# Perform Agglomerative Clustering with 4 clusters
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds['Clusters'] = yhat_AC
data['Clusters'] = yhat_AC

# Visualize the clusters in a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='3d', label='bla')
ax.scatter(x, y, z, s=40, c=PCA_ds['Clusters'], marker='o')
ax.set_title('The plot of clusters')
plt.show()

# Evaluating Models

# Visualize the distribution of clusters
pal = ['#682F2F', '#B9C0C9', '#9F8A78', '#F3AB60']
pl = sns.countplot(x=data['Clusters'], palette=pal)
pl.set_title('Distribution of the clusters')
plt.show()

# Visualize the clusters based on income and spending
pl = sns.scatterplot(data=data, x=data['Spent'], y=data['Income'], hue=data['Clusters'], palette=pal)
pl.set_title('Clusters profile based on income and spending')
plt.legend()
plt.show()

# Visualize the distribution of spending across clusters
plt.figure()
pl = sns.swarmplot(x=data['Clusters'], y=data['Spent'], color='#CBEDDD', alpha=0.5)
pl = sns.boxenplot(x=data['Clusters'], y=data['Spent'], palette=pal)
plt.show()

# Calculate the total number of promotions accepted
data['Total_Promos'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data[
    'AcceptedCmp5']

# Visualize the count of promotions accepted across clusters
plt.figure()
pl = sns.countplot(x=data['Total_Promos'], hue=data['Clusters'], palette=pal)
pl.set_title('Count of promotions accepted')
pl.set_xlabel('Number of total accepted promotions')
plt.show()

# Visualize the number of deals purchased across clusters
plt.figure()
pl = sns.boxenplot(y=data['NumDealsPurchases'], x=data['Clusters'], palette=pal)
pl.set_title('Number of deals purchased')
plt.show()

# Profiling

personal = ['Kidhome', 'Teenhome', 'Customer_For', 'Age', 'Children', 'Family_Size', 'Is_Parent', 'Education',
            'Living_With']

# Visualize the relationships between personal attributes and spending across clusters
for i in personal:
    plt.figure()
    sns.jointplot(x=data[i], y=data['Spent'], hue=data['Clusters'], kind='kde', palette=pal)
    plt.show()

# Profile of Cluster 0:
# - Mostly parents
# - Family size ranges from 2 to 4 members
# - Majority have teenage kids
# - Relatively older
#
# Profile of Cluster 1:
# - Majority are parents
# - Family size is mostly 3 members
# - Majority have non-teenage kids
# - Relatively younger
#
# Profile of Cluster 2:
# - Not parents
# - Family size ranges from 1 to 2 members
# - Mostly couples, some singles
# - Span all ages
# - High income group
#
# Profile of Cluster 3:
# - Mostly parents
# - Family size ranges from 2 to 5 members
# - Majority have teenage kids
# - Relatively older
# - Lower income group


# In[ ]:





# In[ ]:




