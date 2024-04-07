#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('float_format', '{:f}'.format)

import matplotlib.pyplot as plt
import matplotlib.colors as colors  
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import metrics 
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import  RobustScaler
from sklearn.preprocessing import PowerTransformer


# In[3]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[4]:


column_list = (list(df.columns))
print(column_list)


# In[7]:


df.isnull().sum()


# Spliting the data into train and test 

# In[8]:


'''This code splits a dataset into two sections: y contains the target variable, 
representing the class labels to predict, extracted from the "Class" column,
while X holds the feature variables, excluding "Class". 
This separation sets the stage for training machine learning models by organizing the input data and target labels.
'''
y= df["Class"]
X = df.drop("Class", axis = 1)
y.shape,X.shape


# In[23]:


'''
This code separates the dataset into training and testing subsets, allocating 80% for training (X_train and y_train) 
and 20% for testing (X_test and y_test). The train_test_split()function from scikit-learn accomplishes this,
preserving the original class distribution for both sets. The chosen random stateensures reproducibility,
while the shapes of the resulting sets provide insight into their sizes for subsequent model training and evaluation.
'''
# Spltting the into 80:20 train test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42,stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[10]:


'''This code snippet checks the count of fraudulent instances in the original dataset and its training and testing subsets. 
Comparing these counts ensures consistencyin class distribution after the split, 
crucial for maintaining dataset representativeness during model training and evaluation.'''
# Checking the split of the class label
print(" Fraudulent Count for Full data : ",np.sum(y))
print("Fraudulent Count for Train data : ",np.sum(y_train))
print(" Fraudulent Count for Test data : ",np.sum(y_test))


# In[11]:


# Save the testing set for evaluation
X_test_saved = X_test.copy()
y_test_saved = y_test.copy()
print("Saved X_test & y_test")


# In[13]:


# As PCA is already performed on the dataset from V1 to V28 features, we are scaling only Amount field
scaler = RobustScaler()

# Scaling the train data
X_train[["Amount"]] = scaler.fit_transform(X_train[["Amount"]])

# Transforming the test data
X_test[["Amount"]] = scaler.transform(X_test[["Amount"]])


# In[14]:


X_train.head()


# In[15]:


X_test.head()


# Checking Skewness

# In[25]:


'''This code snippet employs Seaborn's histplot function to visualize the distribution of values across different features in the training dataset.
Each feature is represented by a histogram subplot within a grid layout, facilitating a detailed examination
of its distribution characteristics, including central tendency, spread, and skewness.'''

var = X_train.columns

with plt.style.context('dark_background'):
    fig, axes = plt.subplots(10, 3, figsize=(30, 45), facecolor='navy')
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(var):
            sns.histplot(X_train[var[i]], ax=ax)
            ax.set_title(var[i], fontsize=20) 
            ax.set_ylabel("Count", fontsize=20)  # set ylabel of the subplot
            ax.tick_params(axis='both', labelsize=15) 
            ax.set_xlabel('') # set empty string as x label of the subplot

    plt.tight_layout()
    plt.show()


#  Many features tend to be highly skewed, meaning they have an asymmetrical distribution. To address this, we will check the skewness using the skew() function. If the skewness falls outside the range of -1 to 1, we will use a power transform to normalize the data.

# In[26]:


'''This code snippet systematically assesses the skewness of features within the training dataset (X_train). 
By iterating through each feature and calculating its skewness using the skew() function,
the code generates a list of skewness values. These values are then organized into a DataFrame (tmp),
where each feature's name is paired with its corresponding skewness measurement
Skewness serves as a crucial indicator of the distributional properties of the data, informing decisions regarding data preprocessing and 
transformation techniques. For instance, features exhibiting substantial skewness may require adjustments such as log transformations or
scaling to mitigate their impact on downstream modeling tasks. This systematic approach to assessing skewness
facilitates comprehensive data exploration and prepares the data appropriately for subsequent analysis and modeling endeavors.'''

var = X_train.columns
skew_list = []
for i in var:
    skew_list.append(X_train[i].skew())

tmp = pd.concat([pd.DataFrame(var, columns=["Features"]), pd.DataFrame(skew_list, columns=["Skewness"])], axis=1)
tmp.set_index("Features", inplace=True)
tmp


# There is some skweness present in the Features and we will filter the one having the skewness less than -1 and greateer than 1

# In[28]:


# Filtering the features which has skewness less than -1 and greater than +1
"""
The code finds feature­s with skewness outside the­ range of -1 to 1. These fe­atures might need transformation to boost mode­l performance. 
The list 'ske­wed' holds the names of the­se features.
"""
skewed = tmp.loc[(tmp["Skewness"] > 1) | (tmp["Skewness"] <-1 )].index
skewed.tolist()


# Treating Skewness

# In[29]:


'''This code utilize­s the PowerTransforme­r tool, a fe­ature of scikit-learn. It fine-tune­s the­ layout of information, 
molding it similar to a Gaussian (bell-shape­d) graph. It introduce­s two method­s: Yeo-Johnson and Box-Cox.
They adapt diffe­rent types of data, eve­n those that are negative­. The option copy=True­ helps maintain the­ original dataset,
safeguarding it from unintende­d changes. The transformer initially aligns with the­ training information, then modifies it. 
This paves the­ way for enhanced outcomes with ce­rtain machine-learning algorithms.
Preproce­ssing plays a pivotal part­. It boosts the stability and accuracy of prediction models. It tackle­s uneven distributions.'''

pt= preprocessing.PowerTransformer(method='yeo-johnson', copy=True)  # creates an instance of the PowerTransformer class.
pt.fit(X_train)

X_train_pt = pt.transform(X_train)
X_test_pt = pt.transform(X_test)

y_train_pt = y_train
y_test_pt = y_test


# In[21]:


print(X_train_pt.shape)
print(y_train_pt.shape)


# In[27]:


var = X_train.columns
with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(30,45), facecolor='navy') # create figure instance
    # fig.suptitle('Histograms of Variables', fontsize=30) # set main title of the figure
    i=0
    for col in var:
        i += 1
        ax = fig.add_subplot(10,3, i) # create subplot
        sns.histplot(X_train[col], ax=ax) # plot histogram
        ax.set_title(col, fontsize=20) # set title of the subplot
        ax.set_ylabel('Count', fontsize=15) # set ylabel of the subplot
        ax.set_xlabel('') # set empty string as x label of the subplot
    fig.subplots_adjust(hspace=0.5, wspace=0.2) # add horizontal and vertical space between subplots
    plt.show()

