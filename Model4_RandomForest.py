#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib 


# In[ ]:


vars = joblib.load('my_variables.pkl')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# # Model Implementation and Visualization

# In[ ]:


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt="g", cmap="coolwarm", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', color='orange', fontsize=16)
    plt.show()

def RF_model(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier()
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

# In[ ]:


def cross_val(x, y):
    
    rf = RandomForestClassifier()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {'roc_auc': 'roc_auc', 
               'f1': 'f1',
               'accuracy': 'accuracy'}

    cv_results = cross_validate(rf, x, y, cv=skf, scoring=scoring)

    for i in range(skf.get_n_splits()):
        print(f"Fold {i+1}: ROC AUC: {cv_results['test_roc_auc'][i]}, F1 Score: {cv_results['test_f1'][i]}, Accuracy: {cv_results['test_accuracy'][i]}")

    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_accuracy = cv_results['test_accuracy'].mean()

    print("\nMean ROC AUC:", mean_roc_auc)
    print("Mean F1 Score:", mean_f1)
    print("Mean Accuracy:", mean_accuracy)


# # Model Performance After Feature Scaling

# In[ ]:


RF_model(vars['x_train'], vars['x_test'], vars['y_train'], vars['y_test'])


# # Model Performance After Feature Transformation

# In[ ]:


RF_model(vars['x_train_pt'], vars['x_test_pt'], vars['y_train'], vars['y_test'])


# # Model Performance After Implementing LDA

# In[ ]:


RF_model(vars['x_train_lda'], vars['x_test_lda'], vars['y_train'], vars['y_test'])


# # Model Performance With Cross Validation

# In[ ]:


cross_val(vars['x'], vars['y'])


# # Model Performance After Implementing SMOTE

# In[ ]:


RF_model(vars['x_smote'], vars['x_test'], vars['y_smote'], vars['y_test'])


# # Model Performance After Implementing ADASYN

# In[ ]:


RF_model(vars['x_adasyn'], vars['x_test'], vars['y_adasyn'], vars['y_test'])

