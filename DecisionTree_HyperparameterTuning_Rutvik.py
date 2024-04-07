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

# In[4]:


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt="g", cmap="coolwarm", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', color='orange', fontsize=16)
    plt.show()

def DT_model(x_train, x_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    train_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    
    print("Classification Report for Training Set:\n")
    print(classification_report(y_train, train_predictions))
    
    print(f"Accuracy Score for Training Set: {train_accuracy}\n")
    
    print(f"ROC AUC Score for Training Set: {train_roc_auc}\n")

    print("Confusion Matrix for Training Set:\n")
    cm_train = confusion_matrix(y_train, train_predictions)
    plot_confusion_matrix(cm_train, labels=['Non-Fraud', 'Fraud'])
    
    print("\nClassification Report for Test Set:\n")
    print(classification_report(y_test, test_predictions))
    
    print(f"Accuracy Score for Test Set: {test_accuracy}\n")
    
    print(f"ROC AUC Score for Test Set: {test_roc_auc}\n")
    
    print("Confusion Matrix for Test Set:")
    cm_test = confusion_matrix(y_test, test_predictions)
    plot_confusion_matrix(cm_test, labels=['Non-Fraud', 'Fraud'])


# # Model Cross Validation

# In[5]:


def cross_val(x, y):
    
    dt = DecisionTreeClassifier()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {'roc_auc': 'roc_auc', 
               'f1': 'f1',
               'accuracy': 'accuracy'}

    cv_results = cross_validate(dt, x, y, cv=skf, scoring=scoring)

    for i in range(skf.get_n_splits()):
        print(f"Fold {i+1}: ROC AUC: {cv_results['test_roc_auc'][i]}, F1 Score: {cv_results['test_f1'][i]}, Accuracy: {cv_results['test_accuracy'][i]}")

    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_accuracy = cv_results['test_accuracy'].mean()

    print("\nMean ROC AUC:", mean_roc_auc)
    print("Mean F1 Score:", mean_f1)
    print("Mean Accuracy:", mean_accuracy)


# # Hyperparameter Tuning

# In[ ]:


def hyperparameter_tuning(x_train, x_test, y_train, y_test):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt = DecisionTreeClassifier()

    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy')

    grid_search.fit(x_train, y_train) 

    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)  
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", roc_auc)

    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)


# # Model Performance After Feature Scaling

# In[6]:


DT_model(vars['x_train'], vars['x_test'], vars['y_train'], vars['y_test'])


# # Model Performance After Feature Transformation

# In[8]:


DT_model(vars['x_train_pt'], vars['x_test_pt'], vars['y_train'], vars['y_test'])


# # Model Performance After Implementing LDA

# In[9]:


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


# # Model Performance With Scaled Features And Hyperparameter Tuning

# In[ ]:


hyperparameter_tuning(vars['x_train'], vars['x_test'], vars['y_train'], vars['y_test'])


# # Model Performance After Implementing SMOTE And Hyperparameter Tuning

# In[ ]:


hyperparameter_tuning(vars['x_smote'], vars['x_test'], vars['y_smote'], vars['y_test'])


# # Model Performance After Implementing ADASYN And Hyperparameter Tuning

# In[ ]:


hyperparameter_tuning(vars['x_adasyn'], vars['x_test'], vars['y_adasyn'], vars['y_test'])

