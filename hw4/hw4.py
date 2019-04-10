import random
import pandas as pd  
import numpy as np
import datetime
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold

def MLP_constructor(sizes):
	#sizes[0] = input_size
	#sizes[-1] = output_size
	weights = {}
	biaes = {}
	for i in range(len(sizes) - 1):
		weights["w" + str(i+1)] = tf.Variable(tf.random_normal([sizes[i], sizes[i+1]]))
		biaes["b" + str(i+1)] = tf.Variable(tf.random_normal([sizes[i+1]]))
	return weights, biaes

def MLP(x, weights, biaes):
	out_layer = x
	for i in range(len(biaes) - 1):
		out_layer = tf.add(tf.matmul(out_layer, weights["w" + str(i+1)]), biaes["b" + str(i+1)])
		out_layer = tf.nn.relu(out_layer)
	i = i + 1
	out_layer = tf.matmul(out_layer, weights["w" + str(i+1)]) + biaes["b" + str(i+1)]
	#out_layer = tf.sigmoid(out_layer)
	return out_layer

def run_MLP(X_train, X_test, Y_train, Y_test, epochs=100, print_info = False):
	learning_rate = 0.001
	training_epochs = epochs
	display_step = 10
	n_input = X_train.shape[1]
	n_classes = Y_train.shape[1]
	X = tf.placeholder("float", [None, n_input])
	Y = tf.placeholder("float", [None, n_classes])
	weights, biaes = MLP_constructor([n_input, 512, 512, 256, 64, n_classes])
	logits = MLP(X, weights, biaes)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)
	# Initializing the variables
	init = tf.global_variables_initializer()

	pred = tf.nn.softmax(logits)  # Apply softmax to logits
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	with tf.Session() as sess:
		sess.run(init)
		train_cost = []
		test_acc = []

		# Training cycle
		for epoch in range(training_epochs):
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: Y_train})
			train_cost.append(c)

			if print_info and (epoch % display_step == 0) :
				print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(c))
				acc = accuracy.eval({X:X_test, Y:Y_test})
				test_acc.append(acc)
				print("Accuracy:", acc)

		if print_info:
			print("Optimization Finished!")
			plt.subplot(2, 1, 1)
			plt.plot(train_cost)
			plt.xlabel("training epoch")
			plt.ylabel("cost")
			plt.subplot(2, 1, 2)
			plt.plot(test_acc)
			plt.xlabel("training epoch * 10")
			plt.ylabel("accuracy")
			plt.show()

		# Test model
		acc = accuracy.eval({X:X_test, Y:Y_test}) 
		print("Accuracy:", acc)
		return acc
def FS_MLP(X_train, X_test, Y_train, Y_test, epochs = 100):
	start = datetime.datetime.now()
	run_MLP(X_train, X_test, Y_train, Y_test, epochs, print_info = True)
	end = datetime.datetime.now()
	print ("running time: " + str(end-start))

def cv1(X, y, n_feature):
	#do gene selection with
	print("Start feature selection from a SVC model...")
	start = datetime.datetime.now()
	X_new = SelectKBest(chi2, k=n_feature).fit_transform(X, y)
	end = datetime.datetime.now()
	selection_time = (end - start).total_seconds()
	print("Finish feature selection!")

	kf = KFold(n_splits=5)
	kf.get_n_splits(X_new)

	acc_list = []
	cv_time = 0
	
	#start CV
	i = 1
	for train_index, test_index in kf.split(X_new):
		print("testing %d fold..." % i)
		i = i + 1
		X_train, X_test = X_new[train_index], X_new[test_index]
		y_train, y_test = y[train_index], y[test_index]
		start = datetime.datetime.now()
		acc = run_MLP(X_train, X_test, y_train, y_test)
		end = datetime.datetime.now()
		acc_list.append(acc)
		cv_time = cv_time + (end - start).total_seconds()
	return [np.mean(acc_list), selection_time, cv_time]

def cv2(X, y, n_feature):
	#LOO
	kf = KFold(n_splits=5)
	kf.get_n_splits(X)

	selection_time = 0
	cv_time = 0
	acc_list = []

	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#feature selection on the training set
		start = datetime.datetime.now()
		selector = SelectKBest(chi2, k=n_feature)
		end = datetime.datetime.now()
		selection_time = selection_time + (end - start).total_seconds()

		X_train_new = selector.fit_transform(X_train, y_train)
		X_test_new = selector.transform(X_test)

		start = datetime.datetime.now()
		acc = run_MLP(X_train_new, X_test_new, y_train, y_test)
		end = datetime.datetime.now()
		cv_time = cv_time + (end - start).total_seconds()
		acc_list.append(acc)
	return [np.mean(acc_list), selection_time, cv_time]

def main():
	#load data
	embryonic_data_no_info_genes = pd.read_csv('./embryonic_data_no_info_genes.txt', sep='\t', index_col=0)
	embryonic_data_no_info_genes.columns = range(embryonic_data_no_info_genes.shape[1])
	label = pd.read_csv('./labels.txt', header = None)

	#seperate the cell according to the date
	seperated_genes = [[],[],[],[],[]]
	for i in range(label.shape[0]):
		seperated_genes[label[0][i] - 3].append(embryonic_data_no_info_genes[:][i])

	#preprocess data for later training
	X = seperated_genes[3] + seperated_genes[4]
	y = np.concatenate((np.zeros(len(seperated_genes[3])), np.ones(len(seperated_genes[4]))))
	y_one_hot = np.concatenate(([[1,0]] * len(seperated_genes[3]), [[0,1]] * len(seperated_genes[4])))
	index = [i for i in range(len(X))]
	
	random.shuffle(index)
	X = np.array(X)
	X = X[index, :]
	y_one_hot = y_one_hot[index,:]

	#test feature selection algorithm
	'''
	print("training without feature selection")
	X_train, X_test, Y_train, Y_test = train_test_split(X, y_one_hot)
	FS_MLP(X_train, X_test, Y_train, Y_test, 200)
	print("press enter to continue...")
	input()

	print("Feature selection with VarianceThreshold.")
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	X_new = sel.fit_transform(X)
	X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(X_new, y_one_hot)
	FS_MLP(X_new_train, X_new_test, Y_new_train, Y_new_test, 200)
	print("press enter to continue...")
	input()

	print("Feature selection with SelectKBest, k = 500")
	X_new = SelectKBest(chi2, k=500).fit_transform(X, y_one_hot)
	X_train, X_test, Y_train, Y_test = train_test_split(X, y_one_hot)
	X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(X_new, y_one_hot)
	FS_MLP(X_new_train, X_new_test, Y_new_train, Y_new_test, 200)
	print("press enter to continue...")
	input()

	print("Feature selection with SelectFromModel, model = SVC")
	svclassifier_linear = SVC(kernel='linear', gamma='scale', C = 10)
	svclassifier_linear.fit(X, y)#can not use one_hot label
	model = SelectFromModel(svclassifier_linear, prefit=True)
	X_new = model.transform(X)
	X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(X_new, y_one_hot)
	FS_MLP(X_new_train, X_new_test, Y_new_train, Y_new_test, 200)
	print("press enter to continue...")
	input()
	

	print("Feature selection with ExtraTreesClassifier")
	clf = ExtraTreesClassifier(n_estimators=50)
	clf = clf.fit(X, y)
	model = SelectFromModel(clf, prefit=True)
	X_new = model.transform(X)
	X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(X_new, y_one_hot)
	FS_MLP(X_new_train, X_new_test, Y_new_train, Y_new_test, 200)
	print("press enter to continue...")
	input()
	'''

	#cv1
	n_features = [20, 100, 200, 500, 1000]
	print("press enter to start cv1")
	input()
	print("Running cv1...")
	cv1_acc_list = []
	cv1_selection_time_list = []
	cv1_cv_time_list = []
	for i in n_features:
		print("Start %d n_feature..." % i)
		[acc, selection_time, cv_time] = cv1(X,y_one_hot,i)
		cv1_acc_list.append(acc)
		cv1_selection_time_list.append(selection_time)
		cv1_cv_time_list.append(cv_time)
	print("accuracy: " + str(cv1_acc_list))
	print("selection time: " + str(cv1_selection_time_list))
	print("cross validation time: " + str(cv1_cv_time_list))

	#cv2
	print("press enter to start cv2")
	input()
	print("Running cv2...")
	cv2_acc_list = []
	cv2_selection_time_list = []
	cv2_cv_time_list = []
	for i in n_features:
		print("Start %d n_feature..." % i)
		[acc, selection_time, cv_time] = cv2(X,y_one_hot,i)
		cv2_acc_list.append(acc)
		cv2_selection_time_list.append(selection_time)
		cv2_cv_time_list.append(cv_time)
	print("accuracy: " + str(cv2_acc_list))
	print("selection time: " + str(cv2_selection_time_list))
	print("cross validation time: " + str(cv2_cv_time_list))

if __name__ == "__main__": 
	main()
