import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def run_RF(X_train, X_test, Y_train, Y_test, n_estimators = 150, bootstrap = True, criterion = 'entropy', max_depth = None,
          min_impurity_decrease = 0.0):
    train_accuracy = []
    test_accuracy = []
    for i in range(1, n_estimators):
        clf = RandomForestClassifier(n_estimators=i, bootstrap = bootstrap, criterion=criterion, max_depth=max_depth, random_state=0, min_impurity_decrease = min_impurity_decrease)
        clf.fit(X_train, Y_train)
        train_accuracy.append(np.sum(clf.predict(X_train) == Y_train) / Y_train.shape[0])
        test_accuracy.append(np.sum(clf.predict(X_test) == Y_test) / Y_test.shape[0])
    plt.plot(train_accuracy, label='training')
    plt.plot(test_accuracy, label='test')
    plt.legend(loc='lower right')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')


#read data
embryonic_data_10genes = pd.read_csv('./embryonic_data_10genes.txt', sep='\t', index_col=0)
embryonic_data_10genes.columns = range(embryonic_data_10genes.shape[1])
label = pd.read_csv('./labels.txt', header = None)

seperated_genes = [[],[],[],[],[]]
#seperate the cell according to the date
for i in range(label.shape[0]):
    seperated_genes[label[0][i] - 3].append(embryonic_data_10genes[:][i])
X = seperated_genes[0] + seperated_genes[2]

#construct training data and test data
y = np.concatenate((np.zeros(len(seperated_genes[0])), np.ones(len(seperated_genes[2]))))
X_train, X_test, Y_train, Y_test = train_test_split(X, y)

#criterion
run_RF(X_train, X_test, Y_train, Y_test, criterion = 'gini')
run_RF(X_train, X_test, Y_train, Y_test, criterion = "entropy")

#max_depth
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 2)
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 5)
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 8)
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 10)

#bootstrap
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 2, bootstrap = False)
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 5, bootstrap = False)
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 8, bootstrap = False)

#min_impurity_decrease
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 8, min_impurity_decrease = 0.1)
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 8, min_impurity_decrease = 0.05)
run_RF(X_train, X_test, Y_train, Y_test, max_depth = 8, min_impurity_decrease = 0.01)

#relative feature importance
clf = RandomForestClassifier(n_estimators=100, bootstrap = False, criterion = 'entropy', max_depth=2, random_state=0)
clf.fit(X_train, Y_train)
print("relative feature importance: ")
print(clf.feature_importances_)
plt.barh(embryonic_data_10genes.index, clf.feature_importances_)



