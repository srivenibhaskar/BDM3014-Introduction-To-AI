#!/usr/bin/env python
# coding: utf-8
Let's start importing the librarys and looking the data
# In[43]:


# loading libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


# In[15]:


df_credit = pd.read_csv("creditcard.csv")


# In[16]:


df_credit.describe()


# In[17]:


df_credit["Class"].value_counts()


# In[18]:


# Again having a look at the shape of the data to check the number of removed rows
dedup_rows = len(data)
df_credit.shape


# In[20]:


original_rows = len(df_credit)


# In[21]:


# Removing duplicate values 
df_credit.drop_duplicates(subset = None, keep = "first", inplace = True, ignore_index = True)


# In[22]:


# Total rows removed 
print("Total duplicate rows removed : ", original_rows - dedup_rows)

Firstly, I will explore through 3 different columns:
Time
Amount
Class
# In[31]:


# Check data distribution
print("Distribution of Normal(0) and Frauds(1): ")
print(df_credit["Class"].value_counts())

# Create the figure
plt.figure(figsize=(7,5))

# Plot class distribution using seaborn barplot
sns.barplot(x=df_credit["Class"].value_counts().index, y=df_credit["Class"].value_counts(), hue=df_credit["Class"].value_counts().index.map({0: 'Normal', 1: 'Fraud'}))

# Set plot title and labels
plt.title("Class Count", fontsize=18)
plt.xlabel("Is fraud?", fontsize=15)
plt.ylabel("Count", fontsize=15)

# Show the plot
plt.show()

We have a clearly imbalanced data.
It's very common when treating of frauds...

First I will do some explore through the Time and Amount.
Second I will explore the V's Features, that are PCA's

Time Features and some Feature Engineering
As our Time feature are in seconds we will transform it ot minutes and hours to get a better understand of the patterns
# In[32]:


timedelta = pd.to_timedelta(df_credit['Time'], unit='s')
df_credit['Time_min'] = (timedelta.dt.components.minutes).astype(int)
df_credit['Time_hour'] = (timedelta.dt.components.hours).astype(int)


# In[33]:


#Exploring the distribuition by Class types throught hours and minutes
plt.figure(figsize=(12,5))
sns.distplot(df_credit[df_credit['Class'] == 0]["Time_hour"], 
             color='g')
sns.distplot(df_credit[df_credit['Class'] == 1]["Time_hour"], 
             color='r')
plt.title('Fraud x Normal Transactions by Hours', fontsize=17)
plt.xlim([-1,25])
plt.show()


# In[34]:


#Exploring the distribuition by Class types throught hours and minutes
plt.figure(figsize=(12,5))
sns.distplot(df_credit[df_credit['Class'] == 0]["Time_min"], 
             color='g')
sns.distplot(df_credit[df_credit['Class'] == 1]["Time_min"], 
             color='r')
plt.title('Fraud x Normal Transactions by minutes', fontsize=17)
plt.xlim([-1,61])
plt.show()

Interesting distribuition, but don't sounds like a clear pattern of actionLooking the statistics of our Amount class frauds and normal transactions
# In[35]:


#To clearly the data of frauds and no frauds
df_fraud = df_credit[df_credit['Class'] == 1]
df_normal = df_credit[df_credit['Class'] == 0]

print("Fraud transaction statistics")
print(df_fraud["Amount"].describe())
print("\nNormal transaction statistics")
print(df_normal["Amount"].describe())

Interesting.
Using this informations I will filter the values to look for Amount by Class
I will filter the "normal" amounts by 3.000
# In[36]:


#Feature engineering to a better visualization of the values
df_credit['Amount_log'] = np.log(df_credit.Amount + 0.01)


# In[37]:


plt.figure(figsize=(14,6))
#I will explore the Amount by Class and see the distribuition of Amount transactions
plt.subplot(121)
ax = sns.boxplot(x ="Class",y="Amount",
                 data=df_credit)
ax.set_title("Class x Amount", fontsize=20)
ax.set_xlabel("Is Fraud?", fontsize=16)
ax.set_ylabel("Amount(US)", fontsize = 16)

plt.subplot(122)
ax1 = sns.boxplot(x ="Class",y="Amount_log", data=df_credit)
ax1.set_title("Class x Amount", fontsize=20)
ax1.set_xlabel("Is Fraud?", fontsize=16)
ax1.set_ylabel("Amount(Log)", fontsize = 16)

plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()

We can see a slightly difference in log amount of our two Classes.
The IQR of fraudulent transactions are higher than normal transactions, but normal transactions have highest valuesLooking a scatter plot of the Time_min distribuition by Amount
# In[38]:


#Looking the Amount and time distribuition of FRAUD transactions
ax = sns.lmplot(y="Amount", x="Time_min", fit_reg=False,aspect=1.8,
                data=df_credit, hue='Class')
plt.title("Amounts by Minutes of Frauds and Normal Transactions",fontsize=16)
plt.show()

Looking a scatter plot of the Time_hour distribuition by Amoun
# In[39]:


ax = sns.lmplot(y="Amount", x="Time_hour", fit_reg=False,aspect=1.8,
                data=df_credit, hue='Class')
plt.title("Amounts by Hour of Frauds and Normal Transactions", fontsize=16)

plt.show()

I will use boxplot to search differents distribuitions:
We are searching for features that diverges from normal distribuition
# In[49]:


#Looking the V's features
columns = df_credit.iloc[:,1:29].columns

frauds = df_credit.Class == 1
normals = df_credit.Class == 0

grid = gridspec.GridSpec(14, 2)
plt.figure(figsize=(15,20*4))

for n, col in enumerate(df_credit[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(df_credit[col][frauds], bins = 50, color='g') #Will receive the "semi-salmon" violin
    sns.distplot(df_credit[col][normals], bins = 50, color='r') #Will receive the "ocean" color
    ax.set_ylabel('Density')
    ax.set_title(str(col))
    ax.set_xlabel('')
plt.show()


# We can see a interesting different distribuition in some of our features like V4, V9, V16, V17 and a lot more.
# Now let's take a look on time distribuition
Feature selections
# In[45]:


#I will select the variables where fraud class have a interesting behavior and might can help us predict

df_credit = df_credit[["Time_hour","Time_min","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19","V27","Amount","Class"]]

Some Feature Engineering
# In[46]:


df_credit.Amount = np.log(df_credit.Amount + 0.001)


# In[47]:


#Looking the final df
df_credit.head()


# In[48]:


colormap = plt.cm.Greens

plt.figure(figsize=(14,12))

sns.heatmap(df_credit.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap = colormap, linecolor='white', annot=True)
plt.show()


# In[50]:


# Sort correlations with respect to the class variable
class_correlation = corr['Class'].drop('Class').sort_values(ascending=False)

# Get the top 3 correlated variables
top_3_correlated_variables = class_correlation.index[:3]

# Print top 3 correlated variables with their correlation values
print("Top 3 correlated variables with class variable:")
for var in top_3_correlated_variables:
    correlation_value = class_correlation[var]
    print(f"{var}: {correlation_value}")

# Set up the matplotlib figure
plt.figure(figsize=(18, 12))

# Set the font scale for the heatmap
sns.set(font_scale=0.8)

# Create a heatmap using seaborn for top correlated variables including the class variable
top_corr_matrix = data[[*top_3_correlated_variables, 'Class']].corr()
sns.heatmap(top_corr_matrix, cmap='viridis', annot=True)

# Show plot
plt.show()

