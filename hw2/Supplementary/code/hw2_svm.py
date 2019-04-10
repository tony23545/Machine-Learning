import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
import seaborn as sns

class mySVC:
	def __init__(self, kernel, gamma, C, n_components = 2):
		print(kernel + " kernel, gamma = " + str(gamma) + ", C = " + str(C))
		self.svclassifier = SVC(kernel = kernel, gamma = gamma, C = C)
		self.pca = decomposition.PCA(n_components = n_components)
		self.name = str(kernel) + "_svc_" + str(gamma) + "_" + str(C)

	def fit(self, X, y):
		print("Fitting....")
		self.svclassifier.fit(X, y)
		print("Finish fitting.")

	def predict(self, X, y):
		y_pred = self.svclassifier.predict(X)
		print('confusion_matrix: ')
		print(confusion_matrix(y, y_pred))
		print('classification_report: ')
		print(classification_report(y,y_pred))

	def visualize(self, x, y):
		print("Visualizing...")
		pc_all = self.pca.fit_transform(x)
		pc_df_all = pd.DataFrame(data = pc_all , columns = ['PC1_all', 'PC2_all'])
		pc_df_all['Cluster'] = y
		pc_df_all.head()
		sns.lmplot( x="PC1_all", y="PC2_all", data=pc_df_all, fit_reg=False, hue='Cluster',legend=True, scatter_kws={"s": 10})
		#visual support vector, with the same PCA model
		pc_all = self.pca.transform(self.svclassifier.support_vectors_)
		pc_df_sv = pd.DataFrame(data = pc_all , columns = ['PC1_sv', 'PC2_sv'])
		pc_df_sv['Cluster'] = y[self.svclassifier.support_]
		pc_df_sv.head()
		sns.lmplot( x="PC1_sv", y="PC2_sv", data=pc_df_sv, fit_reg=False, hue='Cluster',legend=True, scatter_kws={"s": 10})
		plt.show()

	def record_vs(self):
		f = open(self.name + "_sv" + ".txt", "w")
		f.write(np.array2string(self.svclassifier.support_vectors_))
		f.close

	#automatically run all the experiment related to a SVC
	def autoRun(self, X_train, X_test, Y_train, Y_test):
		self.fit(X_train, Y_train)
		print("Predict on training set")
		self.predict(X_train, Y_train)
		print("Predict on test set")
		self.predict(X_test, Y_test)
		self.visualize(X_train, Y_train)
		self.record_vs()

def main():
	#load training data and labels
	embryonic_data_10genes = pd.read_csv('./embryonic_data_10genes.txt', sep='\t', index_col=0)
	embryonic_data_10genes.columns = range(embryonic_data_10genes.shape[1])
	label = pd.read_csv('./labels.txt', header = None)

	#seperate the data by date
	seperated_genes = [[],[],[],[],[]]
	for i in range(label.shape[0]):
		seperated_genes[label[0][i] - 3].append(embryonic_data_10genes[:][i])

	#construct training set with E3 and E5 data
	X = seperated_genes[0] + seperated_genes[2]
	y = np.concatenate((np.zeros(len(seperated_genes[0])), np.ones(len(seperated_genes[2]))))
	X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size = 0.2) #split rate 0.2

	#test linear kernel of C = 0.001 and C = 100
	linear_svc1 = mySVC('linear', 'scale', 0.001, 2)
	linear_svc1.autoRun(X_train, X_test, Y_train, Y_test)

	linear_svc2 = mySVC('linear', 'scale', 100, 2)
	linear_svc2.autoRun(X_train, X_test, Y_train, Y_test)
	
	#test rbf kernel of gamma = 10e-3, 10e-6, 10e-9
	rbf_svc1 = mySVC('rbf', 0.001, 1.0, 2)
	rbf_svc1.autoRun(X_train, X_test, Y_train, Y_test)

	rbf_svc2 = mySVC('rbf', 0.000001, 1.0, 2)
	rbf_svc2.autoRun(X_train, X_test, Y_train, Y_test)

	rbf_svc3 = mySVC('rbf', 0.000000001, 1.0, 2)
	rbf_svc3.autoRun(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
	main()