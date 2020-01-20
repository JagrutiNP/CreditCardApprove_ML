# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:05:43 2020

@author: 10644430
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import mode
from feature_engine.categorical_encoders import RareLabelCategoricalEncoder
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score,roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint

## read data 
read_data=pd.read_csv("D:\JAGRUTI\Credit_approval\\credit-approval_csv.csv")

## fill missing values in dataset
for col_name in read_data:
    if read_data[col_name].isnull().any()==True:
        if read_data[col_name].dtype=='object':            
            #missing_dict[col_name]=mode(read_data[col_name])
            read_data[col_name].fillna(mode(read_data[col_name]),inplace=True)
        elif read_data[col_name].dtype=='int' or 'float':
            #missing_dict[col_name]=read_data[col_name].median()
            read_data[col_name].fillna(read_data[col_name].median(),inplace=True)

read_data.isnull().mean()

read_data['class'].value_counts().plot(kind='pie')
plt.title('Distribution of Class')
read_data_copy=read_data

A1_analysis=pd.crosstab(read_data['A1'], read_data['class'],margins=True)
A1_plot=pd.crosstab(read_data['A1'], read_data['class']).plot(kind='bar')
plt.xlabel('A1')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A1')

A4_analysis=pd.crosstab(read_data['A4'], read_data['class'],margins=True)
A4_plot=pd.crosstab(read_data['A4'], read_data['class']).plot(kind='bar')
plt.xlabel('A4')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A4')

A5_analysis=pd.crosstab(read_data['A5'], read_data['class'],margins=True)
A5_plot=pd.crosstab(read_data['A5'], read_data['class']).plot(kind='bar')
plt.xlabel('A5')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A5')

A6_analysis=pd.crosstab(read_data['A6'], read_data['class'],margins=True)
A6_plot=pd.crosstab(read_data['A6'], read_data['class']).plot(kind='bar')
plt.xlabel('A6')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A6')

A7_analysis=pd.crosstab(read_data['A7'], read_data['class'],margins=True)
A7_plot=pd.crosstab(read_data['A7'], read_data['class']).plot(kind='bar')
plt.xlabel('A7')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A7')

A9_analysis=pd.crosstab(read_data['A9'], read_data['class'],margins=True)
A9_plot=pd.crosstab(read_data['A9'], read_data['class']).plot(kind='bar')
plt.xlabel('A9')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A9')


A10_analysis=pd.crosstab(read_data['A10'], read_data['class'],margins=True)
A10_plot=pd.crosstab(read_data['A10'], read_data['class']).plot(kind='bar')
plt.xlabel('A10')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A10')

A12_analysis=pd.crosstab(read_data['A12'], read_data['class'],margins=True)
A12_plot=pd.crosstab(read_data['A12'], read_data['class']).plot(kind='bar')
plt.xlabel('A12')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A12')


A13_analysis=pd.crosstab(read_data['A13'], read_data['class'],margins=True)
A13_plot=pd.crosstab(read_data['A13'], read_data['class']).plot(kind='bar')
plt.xlabel('A13')
plt.ylabel('frequency')
plt.title('distribution of classes in different categories of A13')

boxplot_A2=read_data.boxplot(column=["A2"],by=['class'])
boxplot_A3=read_data.boxplot(column=['A3'],by=['class'])
boxplot_A8=read_data.boxplot(column=['A8'],by=['class'])
boxplot_A11=read_data.boxplot(column=['A11'],by=['class'])
boxplot_A14=read_data.boxplot(column=['A14'],by=['class'])
boxplot_A15=read_data.boxplot(column=['A15'],by=['class'])



read_data = read_data[read_data.A4 != 'l']
read_data=read_data[read_data.A5 != 'gg']



## remove category l and gg from variable l A4 and A5 simultaneously
read_data = read_data[read_data.A4 != 'l']
read_data=read_data[read_data.A5 != 'gg']

## encode rare categories in dataset
rare_encoder = RareLabelCategoricalEncoder(
    tol=0.05,  # minimal percentage to be considered non-rare
    n_categories=2, # minimal number of categories the variable should have to re-cgroup rare categories
    variables=['A1','A4','A5','A6','A7','A9','A10','A12'] # variables to re-group
) 
rare_encoder.fit(read_data)
rare_encoder.variables
read_data = rare_encoder.transform(read_data)


rare_encoder_A13 = RareLabelCategoricalEncoder(
    tol=0.1,  # minimal percentage to be considered non-rare
    n_categories=2, # minimal number of categories the variable should have to re-cgroup rare categories
    variables=['A13'] # variables to re-group
) 
rare_encoder_A13.fit(read_data)
rare_encoder_A13.variables
read_data = rare_encoder_A13.transform(read_data)

## standarized numerical variables
read_data[['A2', 'A3','A8','A11','A14','A15']] = StandardScaler().fit_transform(read_data[['A2', 'A3','A8','A11','A14','A15']])

## check if there is relation between independent categorical and dependent categorical variable using chi-square test
chiSquaredResult={}
def check_categorical_dependency(crosstab_table, confidence_interval,var):
    stat, p, dof, expected = stats.chi2_contingency(crosstab_table)
    print ("Chi-Square Statistic value = {}".format(stat))
    print ("P - Value = {}".format(p))
    alpha = 1.0 - confidence_interval
    if p <= alpha:
        chiSquaredResult[var]='Dependent (reject H0)'
    else:
        chiSquaredResult[var]='Independent (fail to reject H0)'
    return chiSquaredResult

categorical_variables=["A1","A12","A4","A5","A6","A7","A9","A10","A13"]

for var in categorical_variables:
    crosstab_table=pd.crosstab(read_data[var], read_data['class'])
    chiSquare=check_categorical_dependency(crosstab_table,0.95,var)

## from chi square test came to know that there is no association between A1 and A2 with dependent variable class
read_data_selectVariables=read_data.drop(['A1','A12'],axis=1)

# convert into dummies variables
data=pd.get_dummies(read_data_selectVariables, prefix=['A4','A5','A6','A7','A9','A10','A13'], columns=['A4','A5','A6','A7','A9','A10','A13'],drop_first=True)

## split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['class'], axis=1), # predictors
    data['class'], # target
    test_size=0.3,
    random_state=0)

## to check accuray of model
def model_generate_reports(y_actual,y_predicted):
    print("Accuracy=", accuracy_score(y_actual,y_predicted))
    print("Precision=", precision_score(y_actual,y_predicted,pos_label='+'))
    print("Recall=",recall_score(y_actual,y_predicted,pos_label='+'))
    print("F1_score=",f1_score(y_actual,y_predicted,pos_label='+'))

## logistic regression
logistic=LogisticRegression()
logistic.fit(X_train,y_train)
#predict_train=clf.predict(X_train)
pred_logistic_test=logistic.predict(X_test)
pred_logistic_train=logistic.predict(X_train)
log_accuracy_train=model_generate_reports(y_train,pred_logistic_train)
log_accuracy_test=model_generate_reports(y_test,pred_logistic_test)
from sklearn.metrics import confusion_matrix
confusion_matrix_log = confusion_matrix(y_test, pred_logistic_test)

## decision tree
decisionTree = DecisionTreeClassifier(random_state=0)
decisionTree.fit(X_train,y_train)
#predict_train=clf.predict(X_train)
pred_tree_test=decisionTree.predict(X_test)
pred_tree_train=decisionTree.predict(X_train)
tree_accuracy_train=model_generate_reports(y_train,pred_tree_train)
tree_accuracy_test=model_generate_reports(y_test,pred_tree_test)
from sklearn.metrics import confusion_matrix
confusion_matrixTree = confusion_matrix(y_test, pred_tree_test)

## randomized search cv for decision tree
param_dist = {"max_depth": [2,3,4,5,6,7,8,9, None], 
              "max_features": [2,4,6,8,10,12,14,16,18,20], 
              "min_samples_leaf":[2,4,6,8,10,12,14,16,18,20,22,24,26], 
              "criterion": ["gini", "entropy"]} 


tree_cv = RandomizedSearchCV(decisionTree, param_dist, cv = 5) 
tree_cv.fit(X_train,y_train)
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_)) 
pred_tree_train=tree_cv.predict(X_train)
pred_tree_test=tree_cv.predict(X_test)
tree_accuracy_train=model_generate_reports(y_train,pred_tree_train)
tree_accuracy_test=model_generate_reports(y_test,pred_tree_test)
confusion_matrixTree=confusion_matrix(y_test, pred_tree_test)

############## final decision tree#################################
tree_algo=DecisionTreeClassifier(random_state=0,min_samples_leaf=10,max_features=20,max_depth=9,criterion='entropy')
tree_algo.fit(X_train,y_train)
#predict_train=clf.predict(X_train)
pred_tree_test=tree_algo.predict(X_test)
pred_tree_train=tree_algo.predict(X_train)
tree_accuracy_train=model_generate_reports(y_train,pred_tree_train)
tree_accuracy_test=model_generate_reports(y_test,pred_tree_test)
tree_algo.feature_importances_
print(dict(zip(X_train.columns, tree_algo.feature_importances_)))


fidt=pd.Series(tree_algo.feature_importances_)
fidt.nlargest(10).plot(kind='barh')
plt.show()
print(fidt.nlargest(10))


clf_forest=RandomForestClassifier(n_jobs=-1)
clf_forest.fit(X_train,y_train)
pred_forest_train=clf_forest.predict(X_train)
pred_forest_test=clf_forest.predict(X_test)
forest_accuracy_train=model_generate_reports(y_train,pred_forest_train)
forest_accuracy_test=model_generate_reports(y_test,pred_forest_test)


param_grid = {
    'n_estimators': randint(100,700),
    'max_features':  randint(2,25),
    'criterion':['gini','entropy'],
    'max_depth':randint(2,30),
    'bootstrap':[False,True],
    'min_samples_leaf':randint(2,30)
}


clf_forest=RandomForestClassifier(n_jobs=-1)
forest_cv = RandomizedSearchCV(clf_forest, param_grid, cv = 5) 
forest_cv.fit(X_train,y_train)
print("Tuned Decision Tree Parameters: {}".format(forest_cv.best_params_)) 
pred_forest_train=forest_cv.predict(X_train)
pred_forest_test=forest_cv.predict(X_test)
forest_accuracy_train=model_generate_reports(y_train,pred_forest_train)
forest_accuracy_test=model_generate_reports(y_test,pred_forest_test)
confusion_matrix_random=confusion_matrix(y_test, pred_forest_test)

################## final random forest algo#########################
forest_algo=RandomForestClassifier(n_jobs=-1,bootstrap=False,criterion='gini', max_depth=14,max_features=16,min_samples_leaf=15,n_estimators= 401)
forest_algo.fit(X_train,y_train)
pred_forest_train=forest_algo.predict(X_train)
pred_forest_test=forest_algo.predict(X_test)
forest_accuracy_train=model_generate_reports(y_train,pred_forest_train)
forest_accuracy_test=model_generate_reports(y_test,pred_forest_test)



