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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# # Model Implementation and Visualization

# In[18]:


def plot_confusion_matrix(cm, labels):
    # Function to plot the confusion matrix
    plt.figure(figsize=(3, 2))  # Set the figure size
    sns.heatmap(cm, annot=True, fmt="g", cmap="coolwarm", xticklabels=labels, yticklabels=labels)  # Plot heatmap with annotations
    plt.xlabel('Predicted')  # Set the label for x-axis
    plt.ylabel('Actual')  # Set the label for y-axis
    plt.title('Confusion Matrix', color='orange', fontsize=16)  # Set the title of the plot
    plt.show()  # Display the plot

def LR_model(x_train, x_test, y_train, y_test):
    # Function to train and evaluate the Logistic Regression model
    model = LogisticRegression()  # Create an instance of Logistic Regression model
    model.fit(x_train, y_train)  # Train the model on the training data

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

# In[19]:


def cross_val(x, y):
    # Function to perform cross-validation
    lr = LogisticRegression()  # Create an instance of Logistic Regression model

    # Define the cross-validation strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define scoring metrics for evaluation
    scoring = {'roc_auc': 'roc_auc', 
               'f1': 'f1',
               'accuracy': 'accuracy'}

    # Perform cross-validation and obtain the evaluation scores
    cv_results = cross_validate(lr, x, y, cv=skf, scoring=scoring)

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


# # Hyperparameter Tuning 

# In[20]:


def hyperparameter_tuning(x_train, x_test, y_train, y_test):
    # Function for hyperparameter tuning
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  
        'penalty': ['l1', 'l2']  
    }
    
    lr = LogisticRegression()  # Create an instance of Logistic Regression model

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)  # Fit the grid search to the training data

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Get the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)  # Make predictions on the test data
    
    # Calculate evaluation metrics on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", roc_auc)

    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)


# # Model Performance After Feature Scaling

# In[21]:


LR_model(vars['x_train'], vars['x_test'], vars['y_train'], vars['y_test'])


# # Model Performance After Feature Transformation

# In[22]:


LR_model(vars['x_train_pt'], vars['x_test_pt'], vars['y_train'], vars['y_test'])


# # Model Performance After Implementing LDA

# In[23]:


LR_model(vars['x_train_lda'], vars['x_test_lda'], vars['y_train'], vars['y_test'])


# # Model Performance With Cross Validation

# In[24]:


cross_val(vars['x'], vars['y'])


# # Model Performance After Implementing SMOTE

# In[13]:


LR_model(vars['x_smote'], vars['x_test'], vars['y_smote'], vars['y_test'])


# # Model Performance After Implementing ADASYN

# In[14]:


LR_model(vars['x_adasyn'], vars['x_test'], vars['y_adasyn'], vars['y_test'])


# # Model Performance With Scaled Features And Hyperparameter Tuning

# In[15]:


hyperparameter_tuning(vars['x_train'], vars['x_test'], vars['y_train'], vars['y_test'])


# # Model Performance After Implementing SMOTE And Hyperparameter Tuning

# In[16]:


hyperparameter_tuning(vars['x_smote'], vars['x_test'], vars['y_smote'], vars['y_test'])


# # Model Performance After Implementing ADASYN And Hyperparameter Tuning

# In[17]:


hyperparameter_tuning(vars['x_adasyn'], vars['x_test'], vars['y_adasyn'], vars['y_test'])

