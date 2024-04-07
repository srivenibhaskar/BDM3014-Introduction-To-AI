#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib 


# In[2]:


vars = joblib.load('my_variables.pkl')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# # Model Implementation and Visualization

# In[13]:


# Function to plot the confusion matrix
def plot_confusion_matrix(cm, labels):
    # Import necessary libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set the size of the plot
    plt.figure(figsize=(3, 2))
    
    # Plot the heatmap
    sns.heatmap(cm, annot=True, fmt="g", cmap="coolwarm", xticklabels=labels, yticklabels=labels)
    
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', color='orange', fontsize=16)
    
    # Display the plot
    plt.show()

# Function to train and evaluate the Decision Tree model
def DT_model(x_train, x_test, y_train, y_test):
    # Import necessary libraries
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
    
    # Create an instance of the Decision Tree classifier
    model = DecisionTreeClassifier()
    
    # Train the model on the training data
    model.fit(x_train, y_train)

    # Make predictions on the training and test data
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    
    # Calculate accuracy scores for training and test sets
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    # Calculate ROC AUC scores for training and test sets
    train_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    
    # Print classification report for training set
    print("Classification Report for Training Set:\n")
    print(classification_report(y_train, train_predictions))
    
    # Print accuracy score for training set
    print(f"Accuracy Score for Training Set: {train_accuracy}\n")
    
    # Print ROC AUC score for training set
    print(f"ROC AUC Score for Training Set: {train_roc_auc}\n")

    # Print confusion matrix for training set
    print("Confusion Matrix for Training Set:\n")
    cm_train = confusion_matrix(y_train, train_predictions)
    plot_confusion_matrix(cm_train, labels=['Non-Fraud', 'Fraud'])
    
    # Print classification report for test set
    print("\nClassification Report for Test Set:\n")
    print(classification_report(y_test, test_predictions))
    
    # Print accuracy score for test set
    print(f"Accuracy Score for Test Set: {test_accuracy}\n")
    
    # Print ROC AUC score for test set
    print(f"ROC AUC Score for Test Set: {test_roc_auc}\n")
    
    # Print confusion matrix for test set
    print("Confusion Matrix for Test Set:")
    cm_test = confusion_matrix(y_test, test_predictions)
    plot_confusion_matrix(cm_test, labels=['Non-Fraud', 'Fraud'])


# # Model Cross Validation

# In[14]:


# Function to perform cross-validation
def cross_val(x, y):
    # Import necessary libraries
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    
    # Create an instance of DecisionTreeClassifier
    dt = DecisionTreeClassifier()

    # Define the cross-validation strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define scoring metrics for evaluation
    scoring = {'roc_auc': 'roc_auc', 
               'f1': 'f1',
               'accuracy': 'accuracy'}

    # Perform cross-validation and obtain the evaluation scores
    cv_results = cross_validate(dt, x, y, cv=skf, scoring=scoring)

    # Print evaluation scores for each fold
    for i in range(skf.get_n_splits()):
        print(f"Fold {i+1}: ROC AUC: {cv_results['test_roc_auc'][i]}, F1 Score: {cv_results['test_f1'][i]}, Accuracy: {cv_results['test_accuracy'][i]}")

    # Calculate mean evaluation scores across all folds
    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_accuracy = cv_results['test_accuracy'].mean()

    # Print mean evaluation scores
    print("\nMean ROC AUC:", mean_roc_auc)
    print("Mean F1 Score:", mean_f1)
    print("Mean Accuracy:", mean_accuracy)


# # Model Performance After Feature Scaling

# In[16]:


DT_model(vars['x_train'], vars['x_test'], vars['y_train'], vars['y_test'])


# # Model Performance After Feature Transformation

# In[17]:


DT_model(vars['x_train_pt'], vars['x_test_pt'], vars['y_train'], vars['y_test'])


# # Model Performance After Implementing LDA

# In[18]:


DT_model(vars['x_train_lda'], vars['x_test_lda'], vars['y_train'], vars['y_test'])


# # Model Performance With Cross Validation

# In[10]:


cross_val(vars['x'], vars['y'])


# # Model Performance After Implementing SMOTE

# In[11]:


DT_model(vars['x_smote'], vars['x_test'], vars['y_smote'], vars['y_test'])


# # Model Performance After Implementing ADASYN

# In[12]:


DT_model(vars['x_adasyn'], vars['x_test'], vars['y_adasyn'], vars['y_test'])

