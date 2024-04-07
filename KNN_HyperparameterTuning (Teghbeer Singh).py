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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# # Model Implementation and Visualization

# In[11]:


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt="g", cmap="coolwarm", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', color='orange', fontsize=16)
    plt.show()

def KNN_model(x_train, x_test, y_train, y_test):
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    # Get the probability estimates for both training and testing sets
    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    
    # Calculate ROC AUC score using the probability estimates
    train_roc_auc = roc_auc_score(y_train, train_prob)
    test_roc_auc = roc_auc_score(y_test, test_prob)
    
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
    
    knn = KNeighborsClassifier()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {'roc_auc': 'roc_auc', 
               'f1': 'f1',
               'accuracy': 'accuracy'}

    cv_results = cross_validate(knn, x, y, cv=skf, scoring=scoring)

    for i in range(skf.get_n_splits()):
        print(f"Fold {i+1}: ROC AUC: {cv_results['test_roc_auc'][i]}, F1 Score: {cv_results['test_f1'][i]}, Accuracy: {cv_results['test_accuracy'][i]}")

    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_accuracy = cv_results['test_accuracy'].mean()

    print("\nMean ROC AUC:", mean_roc_auc)
    print("Mean F1 Score:", mean_f1)
    print("Mean Accuracy:", mean_accuracy)


# # Hyperparameter Tuning 

# In[6]:


def hyperparameter_tuning(x_train, x_test, y_train, y_test):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
    
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')

    grid_search.fit(x_train, y_train) 

    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)  
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    
    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)


# Converting the saved variables into Numpy Arrays so that KNN model will get the right data type  

# In[18]:


x_train = np.array(vars['x_train'])
x_test = np.array(vars['x_test'])
y_train = np.array(vars['y_train'])
y_test = np.array(vars['y_test'])
x_train_pt = np.array(vars['x_train_pt'])
x_test_pt = np.array(vars['x_test_pt'])
x_train_lda = np.array(vars['x_train_lda'])
x_test_lda = np.array(vars['x_test_lda'])
x = np.array(vars['x'])
y = np.array(vars['y'])
x_smote = np.array(vars['x_smote'])
y_smote = np.array(vars['y_smote'])
x_adasyn = np.array(vars['x_adasyn'])
y_adasyn = np.array(vars['y_adasyn']) 


# # Model Performance After Feature Scaling

# In[15]:


KNN_model(x_train, x_test, y_train, y_test)


# # Model Performance After Feature Transformation

# In[ ]:


KNN_model(x_train_pt, x_test_pt, y_train, y_test)


# # Model Performance After Implementing LDA

# In[21]:


KNN_model(x_train_lda, x_test_lda, y_train, y_test)


# # Model Performance With Cross Validation

# In[ ]:


cross_val(x, y)


# # Model Performance After Implementing SMOTE

# In[ ]:


KNN_model(x_smote, x_test, y_smote, y_test)


# # Model Performance After Implementing ADASYN

# In[ ]:


KNN_model(x_adasyn, x_test, y_adasyn, y_test)


# # Model Performance With Scaled Features And Hyperparameter Tuning

# In[ ]:


hyperparameter_tuning(x_train, x_test, y_train, y_test)


# # Model Performance After Implementing SMOTE And Hyperparameter Tuning

# In[ ]:


hyperparameter_tuning(x_smote, x_test, y_smote, y_test)


# # Model Performance After Implementing ADASYN And Hyperparameter Tuning

# In[ ]:


hyperparameter_tuning(x_adasyn, x_test, y_adasyn, y_test)

