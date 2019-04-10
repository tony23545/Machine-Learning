#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression

#read in data
embryonic_data_10genes = pd.read_csv('./embryonic_data_10genes.txt', sep='\t', index_col=0)
label = pd.read_csv('./labels.txt', header = None)

#change to column header to integer
embryonic_data_10genes.columns = range(embryonic_data_10genes.shape[1])

#group the gene by day
seperated_genes = [[],[],[],[],[]]
for i in range(label.shape[0]):
    seperated_genes[label[0][i] - 3].append(embryonic_data_10genes[:][i])

'''
Task 1 (logistic regression)
'''
#construct trainning data, E3 and E5
X = seperated_genes[0] + seperated_genes[2]
y = np.concatenate((np.zeros(len(seperated_genes[0])), np.ones(len(seperated_genes[2]))))

#train logistic regression model
clf_LR = LogisticRegressionCV(cv=10, random_state=0, multi_class='multinomial', max_iter = 500).fit(X, y)

#performance on the trainning data
pred_3 = np.sum(clf_LR.predict(seperated_genes[0]))/len(seperated_genes[0])
pred_5 = np.sum(clf_LR.predict(seperated_genes[2]))/len(seperated_genes[2])
print('prediction on E3: ' + str(pred_3) + '\nprediction on E5: ' + str(pred_5))

#apply the logistic regression model to the dataset
fig = plt.figure()
predict = []
for i in range(len(seperated_genes)):
    predict.append(np.sum(clf_LR.predict(seperated_genes[i]))/len(seperated_genes[i]))
y_name = ['E3', 'E4', 'E5', 'E6', 'E7']
plt.subplot(2, 2, 1)
plt.plot(y_name, predict)
plt.title('logistic regrassion classifier applied on 5 days')

#relative importance of logistic regression model
feature_importance = np.abs(clf_LR.coef_)
feature_importance = 100 * feature_importance / np.max(feature_importance)
sorted_idx = np.argsort(feature_importance[0])
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(2, 2, 3)
plt.barh(pos, feature_importance[0][sorted_idx], align='center')
plt.yticks(pos, embryonic_data_10genes.index[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('LR Relative Importance')

'''Task 1 (FLD method)
'''
clf_LDA = LDA().fit(X,y)

#apply the LDA model to the dataset
predict = []
for i in range(len(seperated_genes)):
    predict.append(np.sum(clf_LDA.predict(seperated_genes[i]))/len(seperated_genes[i]))
y_name = ['E3', 'E4', 'E5', 'E6', 'E7']
plt.subplot(2, 2, 2)
plt.plot(y_name, predict)
plt.title('LDA classifier applied on 5 days')

#relative importance of LDA model
feature_importance = np.abs(clf_LDA.coef_)
feature_importance = 100 * feature_importance / np.max(feature_importance)
sorted_idx = np.argsort(feature_importance[0])
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(2, 2, 4)
plt.barh(pos, feature_importance[0][sorted_idx], align='center')
plt.yticks(pos, embryonic_data_10genes.index[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('LDA Relative Importance')

'''
Task 2. linear regression
'''
reg = LinearRegression().fit(embryonic_data_10genes.values.T, label)

#evaluate the linear model
score = reg.score(embryonic_data_10genes.values.T, label)
print('R^2 of the linear model: ' + str(score))

print('coefficient of the linear model: ' + str(reg.coef_))

print('All the tasks are finished!')

plt.show()