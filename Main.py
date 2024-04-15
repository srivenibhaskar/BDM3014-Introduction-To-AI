#!/usr/bin/env python
# coding: utf-8

# # Data Exploration and Data Cleaning

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


credit_df = pd.read_csv('creditcard_dataset.csv')
credit_df.head()


# In[3]:


credit_df.shape


# In[4]:


credit_df.isnull().sum()


# In[5]:


credit_df.info()


# In[6]:


credit_df.describe()


# In[7]:


classes = credit_df['Class'].value_counts()
normal_transaction = round(classes[0]/credit_df['Class'].count()*100,2)
fraud_transaction = round(classes[1]/credit_df['Class'].count()*100, 2)
print(f"Non-Fraudulent : {normal_transaction}%")
print(f"Fraudulent : {fraud_transaction}%")


# In[8]:


plt.figure(figsize=(20,6))

fig, ax = plt.subplots(1, 2, figsize=(18,4))

classes.plot(kind='bar', rot=0, ax=ax[0])
ax[0].set_title('Number of Class Distributions\n(0: No Fraud | 1: Fraud)')
ax[0].set_ylabel('Number of Transactions')
ax[0].set_xlabel('Class')

for container in ax[0].containers:
    ax[0].bar_label(container, label_type='edge', labels=container.datavalues.astype(int))

perc = (classes / credit_df['Class'].count()) * 100

perc.plot(kind='bar', rot=0, ax=ax[1])
ax[1].set_title('Percentage of Distributions\n(0: No Fraud | 1: Fraud)')
ax[1].set_ylabel('Percentage of Transactions')
ax[1].set_xlabel('Class')

for container in ax[1].containers:
    ax[1].bar_label(container, label_type='edge', labels=[f'{round(val,2)}%' for val in container.datavalues])

plt.show()


# * The dataset has very high class imbalance. Only 492 records are there among 284807 records which are labeld as fradudulent transaction.
# * The percentage of distribution for majority class is 99.83% and minority class is 0.17%

# In[9]:


import matplotlib.colors as colors 

cmap = colors.ListedColormap(['blue', 'red'])

plt.figure(figsize=(15,6))

plt.scatter(x=credit_df["Amount"], y=credit_df["Class"], c=credit_df["Class"], cmap=cmap)

plt.colorbar(ticks=[0, 1], orientation='horizontal', aspect=30, pad = 0.15)
plt.title("Amount vs Class scatter plot", fontsize=25)
plt.ylabel("Class", fontsize=15)
plt.xlabel("Amount", fontsize=15)
plt.grid()

legend_elements = [plt.scatter([],[], marker='o', color='blue', label='0 - Non Fraudulent'),
                   plt.scatter([],[], marker='o', color='red', label='1 - Fraudulent')]
plt.legend(handles=legend_elements, loc='upper center')

plt.show()


# Clearly low amount transactions are more likely to be fraudulent than high amount transaction.

# In[10]:


correlation_matrix = credit_df.corr()

plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[11]:


correlation_with_class = credit_df.corrwith(credit_df['Class']).abs()

correlation_with_class = correlation_with_class.drop(index='Class')

top_positive_correlated_features = correlation_with_class.nlargest(5)

top_negative_correlated_features = correlation_with_class.nsmallest(5)

print("Top 5 positively correlated features with 'Class':")
print(top_positive_correlated_features)

print("\nTop 5 negatively correlated features with 'Class':")
print(top_negative_correlated_features)


# * Positive Correlations: V17, V14, V12 V10 and V16 are positively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.
# * Negative Correlations: V2, V4, V11, and V19 are negatively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.

# In[12]:


feature_names = credit_df.columns.drop('Class')

num_features = len(feature_names)
num_cols = 3  
num_rows = num_features // num_cols  

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    
for i, feature in enumerate(feature_names):
    row = i // num_cols
    col = i % num_cols
    sns.kdeplot(data=credit_df[credit_df['Class'] == 0], x=feature, ax=axes[row, col], label='Class 0', color='blue', linestyle='-')
    sns.kdeplot(data=credit_df[credit_df['Class'] == 1], x=feature, ax=axes[row, col], label='Class 1', color='red', linestyle='--')
    axes[row, col].set_title(feature)

plt.tight_layout()
plt.legend()
plt.show()


# # Feature Engineering

# In[13]:


credit_df.drop('Time', axis=1, inplace=True)


# # 1. Feature Scaling

# In[14]:


from sklearn.preprocessing import RobustScaler
import joblib

scaler = RobustScaler()

scaler.fit(credit_df[['Amount']])

joblib.dump(scaler, 'scaler_val.pkl')

credit_df["scaled_amount"] = scaler.transform(credit_df[["Amount"]])


# In[15]:


credit_df.drop('Amount', axis=1, inplace=True)


# In[16]:


from sklearn.model_selection import train_test_split

x = credit_df.drop('Class', axis=1)
y = credit_df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape 


# In[17]:


joblib.dump({'x': x, 'y': y, 'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, 'my_variables.pkl')


# In[18]:


x_train.head()


# In[19]:


y_train.head()


# # Feature Selection

# In[20]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

feature_importances = rf_model.feature_importances_
feature_importances_percentage = feature_importances * 100
sorted_indices = np.argsort(feature_importances)[::-1]

for i in range(len(sorted_indices)):
    feature_index = sorted_indices[i]
    feature_name = x_train.columns[feature_index]
    print(f"Feature '{feature_name}': Importance {feature_importances_percentage[feature_index]:.2f}%")


# In[21]:


plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_indices)), feature_importances_percentage[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), [x_train.columns[idx] for idx in sorted_indices])
plt.xlabel('Importance (%)')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()


# In[22]:


correlation_matrix = x_train.corr()

plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# In[23]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

sorted_indices = np.argsort(feature_importances)[::-1]

feature_subsets = []
mean_accuracies = []

for i in range(5, len(sorted_indices) + 1):
    features = x_train.columns[sorted_indices[:i]]
    feature_subsets.append(features)
    
    logreg_model = LogisticRegression()
    
    scores = cross_val_score(logreg_model, x_train[features], y_train, cv=10, scoring='accuracy')
    mean_accuracies.append(scores.mean())
    
    print(f"Features: {features},\nMean Accuracy: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")


# In[24]:


top_10_features = x_train.columns[sorted_indices[:10]] 
imp_features = np.append(top_10_features, 'scaled_amount') 

print("Important Features:")
for feature in imp_features:
    print(feature)
    
x_train_impFeatures = x_train[imp_features]
x_test_impFeatures = x_test[imp_features]


# In[25]:


existing_vars = joblib.load('my_variables.pkl')
existing_vars['x_train_impFeatures'] = x_train_impFeatures
existing_vars['x_test_impFeatures'] = x_test_impFeatures

joblib.dump(existing_vars, 'my_variables.pkl')


# In[26]:


feature_names = x_train.columns

num_features = len(feature_names)
num_cols = 3  
num_rows = (num_features + num_cols - 1) // num_cols  

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    
for i, feature in enumerate(feature_names):
    row = i // num_cols
    col = i % num_cols
    sns.histplot(data=x_train, x=feature, ax=axes[row, col])
    axes[row, col].set_title(feature)

plt.tight_layout()
plt.show()


# # 2. Feature Transformation

# In[27]:


from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson', copy=True)
pt.fit(x_train)

x_train_pt = pt.transform(x_train)
x_train_pt_df = pd.DataFrame(x_train_pt, columns=x_train.columns)

x_test_pt = pt.transform(x_test)
x_test_pt_df = pd.DataFrame(x_test_pt, columns=x_test.columns)


# In[28]:


existing_vars = joblib.load('my_variables.pkl')
existing_vars['x_train_pt'] = x_train_pt_df
existing_vars['x_test_pt'] = x_test_pt_df

# Save all the variables together
joblib.dump(existing_vars, 'my_variables.pkl')


# In[29]:


feature_names = x_train_pt_df.columns

num_features = len(feature_names)
num_cols = 3  
num_rows = (num_features + num_cols - 1) // num_cols  

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    
for i, feature in enumerate(feature_names):
    row = i // num_cols
    col = i % num_cols
    sns.histplot(data=x_train_pt_df, x=feature, ax=axes[row, col])
    axes[row, col].set_title(feature)

plt.tight_layout()
plt.show()


# # Implementing LDA

# In[30]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)


# In[31]:


existing_vars = joblib.load('my_variables.pkl')
existing_vars['x_train_lda'] = x_train_lda
existing_vars['x_test_lda'] = x_test_lda

joblib.dump(existing_vars, 'my_variables.pkl')


# # Data Sampling Technique - SMOTE 

# In[32]:


from imblearn.over_sampling import SMOTE, ADASYN

smote = SMOTE(random_state=42)

x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


# In[33]:


existing_vars = joblib.load('my_variables.pkl')
existing_vars['x_train_smote'] = x_train_smote
existing_vars['y_train_smote'] = y_train_smote

joblib.dump(existing_vars, 'my_variables.pkl')


# In[36]:


class_distribution_before = {label: count for label, count in zip(*np.unique(y_train, return_counts=True))}

class_distribution_after = {label: count for label, count in zip(*np.unique(y_train_smote, return_counts=True))}

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(class_distribution_before.keys(), class_distribution_before.values(), color='brown')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.bar(class_distribution_after.keys(), class_distribution_after.values(), color='blue')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# # Data Sampling Technique - ADASYN

# In[37]:


adasyn = ADASYN(random_state=42)

x_train_adasyn, y_train_adasyn = adasyn.fit_resample(x_train, y_train)


# In[38]:


existing_vars = joblib.load('my_variables.pkl')
existing_vars['x_train_adasyn'] = x_train_adasyn
existing_vars['y_train_adasyn'] = y_train_adasyn

joblib.dump(existing_vars, 'my_variables.pkl')


# In[39]:


class_distribution_before = {label: count for label, count in zip(*np.unique(y_train, return_counts=True))}

class_distribution_after = {label: count for label, count in zip(*np.unique(y_train_adasyn, return_counts=True))}

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(class_distribution_before.keys(), class_distribution_before.values(), color='brown')
plt.title('Class Distribution Before ADASYN')
plt.xlabel('Class')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.bar(class_distribution_after.keys(), class_distribution_after.values(), color='blue')
plt.title('Class Distribution After ADASYN')
plt.xlabel('Class')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

