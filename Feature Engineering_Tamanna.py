#!/usr/bin/env python
# coding: utf-8

# # Importing all the necessary  libraries

# In[72]:


import pandas as pd #To handle data 
import numpy as np # For  math 
import seaborn as sns # For visualization
import matplotlib.pyplot as plt # To plot the graphs
import matplotlib.gridspec as gridspec # To do the grid of plots


# In[73]:


#loading the data
df_credit = pd.read_csv("creditcard.csv")


# # Removing the duplicate rows

# In[74]:


original_rows = len(df_credit)


# In[75]:


# Removing duplicate values 
df_credit.drop_duplicates(subset = None, keep = "first", inplace = True, ignore_index=True)


# In[76]:


dedup_rows = len(df_credit)


# In[77]:


# Total rows removed 
print("Total duplicate rows removed : ", original_rows - dedup_rows)


# In[78]:


#looking the how data looks
df_credit.head()


# In[79]:


#looking at the type and searching for null values
df_credit.info()


# In[80]:


# The data is stardarized, I will explore them later
#For now I will look the "normal" columns
df_credit[["Time","Amount","Class"]].describe()


# # Firstly, I will explore through 3 different columns:
# #Time
# #Amount
# #Class

# In[81]:


#Lets start looking at the difference between  Normal and Fraud transactions
print("Distribuition of Normal(0) and Frauds(1): ")
print(df_credit["Class"].value_counts())
plt.figure(figsize=(7,5))
sns.barplot(x=df_credit["Class"].value_counts().index, y=df_credit["Class"].value_counts())
plt.title("Class Count", fontsize=18)
plt.xlabel("Is fraud?", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


# # We have a clearly imbalanced data.
# It's very common when treating of frauds...
# 
# First I will explore through the Time and Amount.
# Then I will explore the V's Features, that are PCA's

# # Time Features and some Feature Engineering. 
# As our Time feature are in seconds we will transform it to minutes and hours to get a better understand of the patterns

# In[82]:


timedelta = pd.to_timedelta(df_credit['Time'], unit='s')
df_credit['Time_min'] = (timedelta.dt.components.minutes).astype(int)
df_credit['Time_hour'] = (timedelta.dt.components.hours).astype(int)


# In[83]:


#Exploring the distribuition by Class types throught hours and minutes
plt.figure(figsize=(12,5))
sns.distplot(df_credit[df_credit['Class'] == 0]["Time_hour"], 
             color='g')
sns.distplot(df_credit[df_credit['Class'] == 1]["Time_hour"], 
             color='r')
plt.title('Fraud x Normal Transactions by Hours', fontsize=17)
plt.xlim([-1,25])
plt.show()


# In[84]:


#Exploring the distribuition by Class types throught hours and minutes
plt.figure(figsize=(12,5))
sns.distplot(df_credit[df_credit['Class'] == 0]["Time_min"], 
             color='g')
sns.distplot(df_credit[df_credit['Class'] == 1]["Time_min"], 
             color='r')
plt.title('Fraud x Normal Transactions by minutes', fontsize=17)
plt.xlim([-1,61])
plt.show()


# #  Interesting distribuition, but dosen't looks like a clear pattern of action

# # Looking the statistics of our Amount class frauds and normal transactions

# In[85]:


#To clearly see the data of frauds and no frauds, we explore the amount feature
df_fraud = df_credit[df_credit['Class'] == 1]
df_normal = df_credit[df_credit['Class'] == 0]

print("Fraud transaction statistics")
print(df_fraud["Amount"].describe())
print("\nNormal transaction statistics")
print(df_normal["Amount"].describe())


# # Using this informations I will filter the values to look for Amount by Class
# 

# In[86]:


#Feature engineering to get a better visualization of the values
df_credit['Amount_log'] = np.log(df_credit.Amount + 0.01)


# In[87]:


plt.figure(figsize=(12,6))
# Exploring the Amount by Class and see the distribuition of Amount transactions
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


# We can see a slight difference in log amount of our two Classes.
# The IQR of fraudulent transactions are higher than normal transactions, but normal transactions have highest values

# # Looking at the scatter plot of the Time_min distribution by Amount

# In[88]:


#Looking at the Amount and time distribuition of FRAUD transactions
ax = sns.lmplot(y="Amount", x="Time_min", fit_reg=False,aspect=1.8,
                data=df_credit, hue='Class')
plt.title("Amounts by Minutes of Frauds and Normal Transactions",fontsize=16)
plt.show()


# # Looking a scatter plot of the Time_hour distribuition by Amount

# In[89]:


ax = sns.lmplot(y="Amount", x="Time_hour", fit_reg=False,aspect=1.8,
                data=df_credit, hue='Class')
plt.title("Amounts by Hour of Frauds and Normal Transactions", fontsize=16)

plt.show()


# # Using boxplot to search differents distribuitions:
# We are searching for features that diverges from normal distribuition

# In[90]:


#Displaying at all the V's features from V1 to V28
columns = df_credit.iloc[:,1:29].columns

frauds = df_credit.Class == 1
normals = df_credit.Class == 0

grid = gridspec.GridSpec(14, 2)
plt.figure(figsize=(15,20*4))

for n, col in enumerate(df_credit[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(df_credit[col][frauds], bins = 50, color='g') # Green represents fraud transactions
    sns.distplot(df_credit[col][normals], bins = 50, color='r') # red represents normal transactions
    ax.set_title(str(col))
    ax.set_xlabel('')
plt.show()


# # Feature selections

# In[95]:


# Selecting the variables where fraud class have a interesting behavior and can help us predict

df_credit = df_credit[["Time_hour","Time_min","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19","V27","Amount","Class"]]


# # Some Feature Engineering

# In[96]:


df_credit.Amount = np.log(df_credit.Amount + 0.001)


# In[97]:


#Looking the final updated df
df_credit.head(15)


# # Displaying Correlation matrix amongst all the features using Heat Map

# In[ ]:


colormap = plt.cm.Greens

plt.figure(figsize=(14,12))

sns.heatmap(df_credit.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap = colormap, linecolor='white', annot=True)
plt.show()


# # Checking the top 3 highest correlated features amongst V1 to V28 with the class feature

# In[102]:


corr = df_credit.corr()
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
top_corr_matrix = df_credit[[*top_3_correlated_variables, 'Class']].corr()
sns.heatmap(top_corr_matrix, cmap='viridis', annot=True)

# Show plot
plt.show()


# In[ ]:




