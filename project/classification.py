#!/usr/bin/env python
# coding: utf-8

# # Third Classifier Train/Test with Only Best Features

# ## Remove Features with little to no correlation to packing

# In[3]:


import pandas as pd
data = pd.read_csv('molecular_crystal_combined_data.csv')
data = data[['Packing','N','O','NofRings','Cyanos','logMolWeight',
          'logC','logNofBonds','logD-A','logRotbonds',
          'Alpha(esu)','Volume(cm^3/mol)']]
data.head()


# In[2]:
# !pip install pandas_profiling


# from pandas_profiling import ProfileReport

# profile = ProfileReport(data, explorative=True, minimal=False)
# try:
#    profile.to_widgets()         # view as widget in Notebook
# except:
#    profile.to_notebook_iframe() # view as html iframe in Notebook
# profile.to_file('dataadded.html')    # save as html file

duplicate= data[data.duplicated()]
print(duplicate)
# ## Remove features that are too highly correlated to other features
# * alpha and volume were very similar. Even though volume appears to have a stronger correlation to packing, it makes more sense from a physical standpoint if alpha is kept
# * logC and logNofBonds were highly correlated, and logC correlated better with packing so I only kept that. 

# In[4]:


X = data[['N','O','NofRings','Cyanos','logMolWeight',
          'logC','logD-A','logRotbonds',
          'Alpha(esu)']]
X.head()


# In[5]:


Y = data[['Packing']]
# y = Y
y = Y.Packing
y = y.replace(-1,0)

print("values", y.value_counts(), '\n')



# In[6]:


# Scale data (0-1)
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = pd.DataFrame(scale.fit_transform(X))
X.columns = ['N','O','NofRings','Cyanos','logMolWeight',
          'logC','logD-A','logRotbonds',
          'Alpha(esu)']
X.head()


# ## Find best features now

# In[13]:


# # Best Features
# from sklearn.feature_selection import SelectKBest
# import matplotlib.pyplot as plt

# best_features = SelectKBest(k=5)
# fit = best_features.fit(X,y)

# plt.figure(figsize=(9,5));
# xlabel = [i for i in X.columns];
# plt.bar(xlabel,fit.scores_);
# plt.xticks(rotation = 80);


# In[7]:




# ## Split the Data into Training and Test Sets

# In[8]:


# Train/ Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100,shuffle=True)


# ## Import Classifiers

# In[16]:


# Import 8 Classifiers

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# In[17]:


ada = AdaBoostClassifier()
lr = LogisticRegression()
# xgb = XGBClassifier()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svc = SVC() # support vector classifier
mlp = MLPClassifier(max_iter = 300)

model_list = [ada,lr,knn,dt,rf,svc,mlp]


# In[18]:


# train classifiers

for model in model_list:
    print("Training ",model)
    model.fit(X_train,y_train)
    print(model,"Trained")
    
# !pip install -U scikit-learn


# In[22]:


# # Open a file and use dump()
# import pickle
# with open('THz_classifiersNEW.pkl', 'wb') as file:
      
#     # A new file will be created
#     pickle.dump(model_list, file)


# In[9]:


# import pickle
  
# # Open the file in binary mode
# with open('THz_classifiersNEW.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     model_list = pickle.load(file)
  
#     print(model_list)


# In[10]:


#!pip install sklearn --upgrade
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
model_names = ['AdaBoostClassifier()','LogisticRegression()','KNeighborsClassifer()','DecisionTreeClassifier',
              'RandomForestClassifier()','SVC()','MPLClassifier()']

plt.figure(figsize=(14,8));
for i,model in enumerate(model_list):
    plt.subplot(2,4,i+1); ax = plt.gca();
    plt.title(model_names[i])
    ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,ax=ax)
    #cmat = confusion_matrix(model,X_test,y_test)
plt.show()



# In[11]:
# # create the visualizer and fit training data
from sklearn.metrics import roc_curve, auc
from sklearn import svm

# # Run classifier
classifier = svm.SVC(probability=True)
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiverrating characteristic example')
plt.legend(loc="lower right")
plt.show()

# test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
# plt.grid()
# # plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
# plt.plot([0,1],[0,1],'g--')
# plt.legend()
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC curve")
# plt.grid(color='black', linestyle='-', linewidth=0.5)
# plt.show()
# print('hi')

# In[12]:

# # Open a file and use dump()
# import pickle
# with open('THz_classifiersBEST.pkl', 'wb') as file:
      
#     # A new file will be created
#     pickle.dump(model_list, file)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



for model in model_list:
    print("Testing ",model)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred, pos_label=0)
    model_recall = recall_score(y_test, y_pred, pos_label=0, average='binary', sample_weight=None, zero_division='warn')
    model_F1 = f1_score(y_test, y_pred, pos_label=0 , average='binary', sample_weight=None, zero_division='warn')
    print(model, 'model_precision', model_precision)
    print(model, 'model_recall', model_recall)
    print(model, 'model_F1', model_F1)
    print(model, 'model_accuracy', model_accuracy)


# In[ ]:


# import pickle
  
# # Open the file in binary mode
# with open('THz_classifiersBEST.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     model_list = pickle.load(file)
  
#     print(model_list)


# kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
# degree = [3,4,5]
# gamma= ['scale', 'auto']
# decision_function_shape = ['ovo', 'ovr']
# probability = [True, False]
# shrinking = [True, False]

# # Create the random grid
# random_grid = {'gamma': gamma,
#                'kernel': kernel,
#                'decision_function_shape': decision_function_shape,

#                'probability': probability,
#                'shrinking': shrinking,
#                'degree': degree}

# print(random_grid)

# svc = SVC()
# cv = KFold(n_splits = 2, shuffle=True, random_state=1)
# search = GridSearchCV(estimator =svc, param_grid= random_grid, verbose=2, cv=cv, n_jobs = -1, refit=True)
# result = search.fit(X_train_val, y_train_val)
# print('best_model')
# best_model= result.best_estimator_