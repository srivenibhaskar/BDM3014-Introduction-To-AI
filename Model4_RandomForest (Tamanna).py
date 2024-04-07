#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the joblib library for efficient model serialization and deserialization.
import joblib 


# In[2]:


# Loading variables from a pickled file named 'my_variables.pkl' using joblib.
vars = joblib.load('my_variables.pkl')


# In[3]:


import numpy as np  # Importing NumPy for numerical computations.
import matplotlib.pyplot as plt  # Importing Matplotlib for data visualization.
import seaborn as sns  # Importing Seaborn for statistical data visualization.
from sklearn.ensemble import RandomForestClassifier  # Importing RandomForestClassifier from scikit-learn for classification tasks.
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV  # Importing modules for cross-validation and hyperparameter tuning.
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score  # Importing modules for model evaluation metrics.
import warnings  # Importing warnings to handle warnings during code execution.
warnings.filterwarnings("ignore")  # Ignoring warnings to prevent them from being displayed during execution.


# # Model Implementation and Visualization

# In[4]:


def plot_confusion_matrix(cm, labels):
    """
    Plotting a confusion matrix.

    Parameters:
    - cm (array-like): Confusion matrix.
    - labels (list): List of labels for the classes.

    Returns:
    - None
    """
    # Creating a new figure with a specified size
    plt.figure(figsize=(3, 2))
    # Plotting a heatmap of the confusion matrix with annotations and a specific colormap
    sns.heatmap(cm, annot=True, fmt="g", cmap="coolwarm", xticklabels=labels, yticklabels=labels)
    # Labelling for x-axis
    plt.xlabel('Predicted')
    # Labelling for y-axis
    plt.ylabel('Actual')
    # Title of the plot
    plt.title('Confusion Matrix', color='orange', fontsize=16)
    # Displaying the plot
    plt.show()

def RF_model(x_train, x_test, y_train, y_test):
    """
    Random Forest classifier model.

    Parameters:
    - x_train (array-like): Training data features.
    - x_test (array-like): Test data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Test data labels.

    Returns:
    - None
    """
    # Creating a Random Forest classifier object
    model = RandomForestClassifier()
    # Training the Random Forest classifier using the training data
    model.fit(x_train, y_train)

    # Generating predictions for the training data
    train_predictions = model.predict(x_train)
    # Generating predictions for the test data
    test_predictions = model.predict(x_test)
    
    # Calculating the accuracy score for the training data
    train_accuracy = accuracy_score(y_train, train_predictions)
    # Calculating the accuracy score for the test data
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    # Calculating the ROC AUC score for the training data
    train_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
    # Calculating the ROC AUC score for the test data
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    
    # Printing the classification report for the training data
    print("Classification Report for Training Set:\n")
    print(classification_report(y_train, train_predictions))
    
    # Printing the accuracy score for the training data
    print(f"Accuracy Score for Training Set: {train_accuracy}\n")
    
    # Printing the ROC AUC score for the training data
    print(f"ROC AUC Score for Training Set: {train_roc_auc}\n")

    # Printing the confusion matrix for the training data
    print("Confusion Matrix for Training Set:\n")
    cm_train = confusion_matrix(y_train, train_predictions)
    plot_confusion_matrix(cm_train, labels=['Non-Fraud', 'Fraud'])
    
    # Printing the classification report for the test data
    print("\nClassification Report for Test Set:\n")
    print(classification_report(y_test, test_predictions))
    
    # Printing the accuracy score for the test data
    print(f"Accuracy Score for Test Set: {test_accuracy}\n")
    
    # Printing the ROC AUC score for the test data
    print(f"ROC AUC Score for Test Set: {test_roc_auc}\n")
    
    # Printing the confusion matrix for the test data
    print("Confusion Matrix for Test Set:")
    cm_test = confusion_matrix(y_test, test_predictions)
    plot_confusion_matrix(cm_test, labels=['Non-Fraud', 'Fraud'])


# # Model Cross Validation

# In[5]:


def cross_val(x, y):
    """
    Perform cross-validation using a Random Forest classifier.

    Parameters:
    - x (array-like): Features of the dataset.
    - y (array-like): Labels of the dataset.

    Returns:
    - None
    """
    # Initializing a Random Forest classifier
    rf = RandomForestClassifier()

    # Initializing Stratified K-Folds cross-validator
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Defining scoring metrics for evaluation
    scoring = {'roc_auc': 'roc_auc', 
               'f1': 'f1',
               'accuracy': 'accuracy'}

    # Performing cross-validation
    cv_results = cross_validate(rf, x, y, cv=skf, scoring=scoring)

    # Printing evaluation metrics for each fold
    for i in range(skf.get_n_splits()):
        print(f"Fold {i+1}: ROC AUC: {cv_results['test_roc_auc'][i]}, F1 Score: {cv_results['test_f1'][i]}, Accuracy: {cv_results['test_accuracy'][i]}")

    # Calculating mean evaluation metrics across all folds
    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_accuracy = cv_results['test_accuracy'].mean()

    # Printing mean evaluation metrics
    print("\nMean ROC AUC:", mean_roc_auc)
    print("Mean F1 Score:", mean_f1)
    print("Mean Accuracy:", mean_accuracy)


# # Hyperparameter Tuning 

# In[6]:


def hyperparameter_tuning(x_train, x_test, y_train, y_test):
    """
    Perform hyperparameter tuning for a Random Forest classifier.

    Parameters:
    - x_train (array-like): Training data features.
    - x_test (array-like): Test data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Test data labels.

    Returns:
    - None
    """
    # Defining the grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    # Initializing a Random Forest classifier
    rf = RandomForestClassifier()

    # Initializing GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fitting the grid search to the training data
    grid_search.fit(x_train, y_train) 

    # Getting the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Getting the best model from the grid search
    best_model = grid_search.best_estimator_
    
    # Making predictions on the test data using the best model
    y_pred = best_model.predict(x_test)  
    
    # Calculating and print the test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    
    # Calculating and print the ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", roc_auc)

    # Calculating and print the F1 score
    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)


# # Model Performance After Feature Scaling

# In[7]:


# Training and evaluating a Random Forest classifier model using the training and test data.
RF_model(vars['x_train'], vars['x_test'], vars['y_train'], vars['y_test'])


# # Model Performance After Feature Transformation

# In[8]:


# Training and evaluating a Random Forest classifier model using the preprocessed training and test data.
RF_model(vars['x_train_pt'], vars['x_test_pt'], vars['y_train'], vars['y_test'])


# # Model Performance After Implementing LDA

# In[9]:


# Training and evaluating a Random Forest classifier model using the training and test data transformed using LDA.
RF_model(vars['x_train_lda'], vars['x_test_lda'], vars['y_train'], vars['y_test'])


# # Model Performance With Cross Validation

# In[10]:


# Performing cross-validation using the features and labels.
cross_val(vars['x'], vars['y'])


# # Model Performance After Implementing SMOTE

# In[11]:


# Training and evaluating a Random Forest classifier model using the SMOTE-resampled training data and original test data.
RF_model(vars['x_smote'], vars['x_test'], vars['y_smote'], vars['y_test'])


# # Model Performance After Implementing ADASYN

# In[12]:


# Training and evaluate a Random Forest classifier model using the ADASYN-resampled training data and original test data.
RF_model(vars['x_adasyn'], vars['x_test'], vars['y_adasyn'], vars['y_test'])


# # Model Performance With Scaled Features And Hyperparameter Tuning

# In[13]:


# Performing hyperparameter tuning for a Random Forest classifier using the  training and test data.
hyperparameter_tuning(vars['x_train'], vars['x_test'], vars['y_train'], vars['y_test'])


# # Model Performance After Implementing SMOTE And Hyperparameter Tuning

# In[ 14]:


# Performing hyperparameter tuning for a Random Forest classifier using the  SMOTE-resampled training data and original test data.
hyperparameter_tuning(vars['x_smote'], vars['x_test'], vars['y_smote'], vars['y_test'])


# # Model Performance After Implementing ADASYN And Hyperparameter Tuning

# In[ 15]:


# Performing hyperparameter tuning for a Random Forest classifier using the  ADASYN-resampled training data and original test data.
hyperparameter_tuning(vars['x_adasyn'], vars['x_test'], vars['y_adasyn'], vars['y_test'])

