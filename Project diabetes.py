#!/usr/bin/env python
# coding: utf-8

# # Predicting diabetes using machine learning

# # Import libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import & read dataset

# In[3]:


diabetes_df = pd.read_csv('diabetes (2).csv')


# In[4]:


diabetes_df.head()


# # Exploratory data analysis(EDA)

# In[5]:


diabetes_df.columns


# # Dataset Information

# In[6]:


diabetes_df.info()


# # Dataset Description

# In[7]:


diabetes_df.describe()


# In[8]:


diabetes_df.describe().T


# # Checking for Null Values

# In[9]:


diabetes_df.isnull()


# In[10]:


diabetes_df.isnull().sum()


# # Handling missing values

# In[11]:


diabetes_df_copy = diabetes_df.copy(deep=True)
diabetes_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# Showing the count of NaNs
print(diabetes_df_copy.isnull().sum())


# # Data visualization

# # Data Distribution Before Imputing Missing Values

# In[12]:


p = diabetes_df.hist(figsize=(20, 20))


# # Imputing Missing Values

# In[13]:


diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace=True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace=True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace=True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace=True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace=True)


# # Data Distribution After Imputing Missing Values

# In[14]:


p = diabetes_df_copy.hist(figsize=(20, 20))


# # Null Count Analysis

# In[15]:


p = msno.bar(diabetes_df)


# In[16]:


color_wheel = {1: "#0392cf", 2: "#7bc043"}
colors = diabetes_df["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_df.Outcome.value_counts())
p = diabetes_df.Outcome.value_counts().plot(kind="bar")


# # Distribution and Outliers of Insulin

# In[17]:


plt.subplot(121)
sns.distplot(diabetes_df['Insulin'])
plt.subplot(122)
diabetes_df['Insulin'].plot.box(figsize=(16, 5))
plt.show()


# # Correlation between all the features

# In[18]:


plt.figure(figsize=(12, 10))
# Using seaborn to create a heatmap for the correlation matrix
p = sns.heatmap(diabetes_df.corr(), annot=True, cmap='RdYlGn')


# In[19]:


plt.figure(figsize=(12, 10))
# Using seaborn to create a heatmap for the correlation matrix
p = sns.heatmap(diabetes_df_copy.corr(), annot=True, cmap='RdYlGn')


# # Scaling the Data

# In[20]:


diabetes_df_copy.head()


# # Applying Standard Scaling

# In[21]:


sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_df_copy.drop(['Outcome'], axis=1)), 

columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

X.head()


# # Exploring the Target Column

# In[22]:


y = diabetes_df_copy.Outcome
y.head()


# # Model Building

# # Splitting the Dataset
# 

# In[23]:


X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)


# # Random Forest

# In[25]:


#Building the model using Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)


# In[26]:


#Check the accuracy of the model on the training dataset
rfc_train = rfc.predict(X_train)
from sklearn import metrics
print("Training Accuracy =", format(metrics.accuracy_score(y_train, rfc_train)))


# In[27]:


#The model is overfitted on the training data. Now, let’s check the accuracy of the test data
predictions = rfc.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test, predictions)))


# In[28]:


#Get the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# # Decision Tree

# In[29]:


#Building the model using a Decision Tree:
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)


# In[30]:


#Make predictions on the testing data:

predictions = dtree.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test, predictions)))


# In[31]:


#Get the classification report and confusion matrix:

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# # XgBoost Classifier

# In[32]:


from xgboost import XGBClassifier

xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)


# In[33]:


#Make predictions on the testing data:

xgb_pred = xgb_model.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test, xgb_pred)))


# In[34]:


#Get the classification report and confusion matrix:

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))


# # Support Vector Machine (SVM)

# In[35]:


from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)


# In[36]:


#Make predictions on the testing data:

svc_pred = svc_model.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test, svc_pred)))


# In[37]:


#Get the classification report and confusion matrix:

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test, svc_pred))


# # Model Performance Comparison

# # Feature Importance

# In[38]:


#Let’s retrieve the feature importances from the Random Forest model:

rfc.feature_importances_


# In[39]:


#We will now plot the feature importances to get a clearer picture:

pd.Series(rfc.feature_importances_, index=X.columns).plot(kind='barh')


# # Saving Model – Random Forest

# In[40]:


import pickle 

# Firstly, we will be using the dump() function to save the model using pickle
saved_model = pickle.dumps(rfc) 

# Then we will be loading that saved model
rfc_from_pickle = pickle.loads(saved_model) 

# Lastly, after loading that model we will use this to make predictions 
rfc_from_pickle.predict(X_test)


# In[41]:


diabetes_df.head()


# In[42]:


rfc.predict([[0, 137, 40, 35, 168, 43.1, 2.228, 33]]) # 4th patient
#According to our model, this patient has diabetes.


# In[43]:


diabetes_df.tail()


# In[44]:


rfc.predict([[10, 101, 76, 48, 180, 32.9, 0.171, 63]]) # 763rd patient
#This patient does not have diabetes.


# # KNN

# In[45]:


train_scores = []
test_scores = []

for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))


# In[46]:


train_scores


# In[47]:


test_scores


# In[51]:


max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max Train Score {} % and k = {}'.format(max_train_score * 100, list(map(lambda x: x + 1, train_scores_ind))))


# In[52]:


max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max Test Score {} % and k = {}'.format(max_test_score * 100, list(map(lambda x: x + 1, test_scores_ind))))


# In[57]:


plt.figure(figsize=(12, 5))
p = sns.lineplot(x=range(1, 15), y=train_scores, marker='*', label='Train Score')
p = sns.lineplot(x=range(1, 15), y=test_scores, marker='o', label='Test Score')


# In[63]:


knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# # Confusion matrix

# In[65]:


y_pred = knn.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[66]:


print(classification_report(y_test,y_pred))


# # ROC-AUC Curve

# In[68]:


from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[69]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('knn(n_neighbors=11) ROC curve')
plt.show()


# In[70]:


roc_auc_score(y_test,y_pred_proba)


# # Implementing GridSearchCV

# In[72]:


from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)

print("Best Score:", knn_cv.best_score_)
print("Best Parameters:", knn_cv.best_params_)


# In[ ]:




