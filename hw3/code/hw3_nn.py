import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import datetime

#call this function to run the k-nearest 
def run_KN(X_train, Y_train, X_test, Y_test, kneigh = 3, algorithm = "ball_tree", weights = 'uniform'):
    test_acc = []
    train_acc = []
    #calculate the running time
    starttime = datetime.datetime.now()
    for i in range(1, kneigh):
        neigh = KNeighborsClassifier(n_neighbors=i, algorithm = algorithm, weights = weights)
        neigh.fit(X_train, Y_train)
        test_acc.append(np.sum(neigh.predict(X_test) == Y_test) / Y_test.shape[0])
        train_acc.append(np.sum(neigh.predict(X_train) == Y_train) / Y_train.shape[0])
    endtime = datetime.datetime.now()
    print ("finish in : " + str(endtime - starttime))
    #plot related
    plt.plot(test_acc, label='test')
    plt.plot(train_acc, label='training')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

def main():
	#load data
	embryonic_data_10genes = pd.read_csv('./embryonic_data_10genes.txt', sep='\t', index_col=0)
	embryonic_data_10genes.columns = range(embryonic_data_10genes.shape[1])
	label = pd.read_csv('./labels.txt', header = None)

	seperated_genes = [[],[],[],[],[]]
	#seperate the cell according to the date
	for i in range(label.shape[0]):
		seperated_genes[label[0][i] - 3].append(embryonic_data_10genes[:][i])

	#construct training data on E3 and E5
	X = seperated_genes[0] + seperated_genes[2]
	y = np.concatenate((np.zeros(len(seperated_genes[0])), np.ones(len(seperated_genes[2]))))

	#split the data into training set and test set
	X_train, X_test, Y_train, Y_test = train_test_split(X, y)

	run_KN(X_train, Y_train, X_test, Y_test, 100, "brute")
	run_KN(X_train, Y_train, X_test, Y_test, 100, "brute", "distance")
	run_KN(X_train, Y_train, X_test, Y_test, 100, "kd_tree")
	run_KN(X_train, Y_train, X_test, Y_test, 100, "kd_tree", "distance")
	run_KN(X_train, Y_train, X_test, Y_test, 100, "ball_tree")
	run_KN(X_train, Y_train, X_test, Y_test, 100, "ball_tree", "distance")

if __name__ == "__main__":
	main()

