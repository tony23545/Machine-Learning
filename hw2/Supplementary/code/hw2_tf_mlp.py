import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#load data and labels
embryonic_data_10genes = pd.read_csv('./embryonic_data_10genes.txt', sep='\t', index_col=0)
embryonic_data_10genes.columns = range(embryonic_data_10genes.shape[1])
label = pd.read_csv('./labels.txt', header = None)

#preprocess data
seperated_genes = [[],[],[],[],[]]
for i in range(label.shape[0]):
    seperated_genes[label[0][i] - 3].append(embryonic_data_10genes[:][i])

#training set and test set for E3 and E5 data
X = seperated_genes[0] + seperated_genes[2]
Y = np.concatenate(([[1,0]] * len(seperated_genes[0]), [[0,1]] * len(seperated_genes[2])))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Parameters
learning_rate = 0.001
training_epochs = 500
display_step = 10

# Network Parameters
n_input = 10
n_classes = 2

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
#tensorflow and MLP
def MLP_constructor(sizes):
    #sizes[0] = input_size
    #sizes[-1] = output_size
    weights = {}
    biaes = {}
    for i in range(len(sizes) - 1):
        weights["w" + str(i+1)] = tf.Variable(tf.random_normal([sizes[i], sizes[i+1]]))
        biaes["b" + str(i+1)] = tf.Variable(tf.random_normal([sizes[i+1]]))
    return weights, biaes

weights, biaes = MLP_constructor([10, 64, 128, 64, 32, 2])

def MLP(x):
    out_layer = x
    for i in range(len(biaes) - 1):
        out_layer = tf.add(tf.matmul(out_layer, weights["w" + str(i+1)]), biaes["b" + str(i+1)])
        out_layer = tf.nn.relu(out_layer)
    i = i + 1
    out_layer = tf.matmul(out_layer, weights["w" + str(i+1)]) + biaes["b" + str(i+1)]
    #out_layer = tf.sigmoid(out_layer)
    return out_layer

# Construct model
logits = MLP(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
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
        _, c = sess.run([train_op, loss_op], feed_dict={X: X_train,
                                                            Y: Y_train})
        train_cost.append(c)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(c))
            
            
            acc = accuracy.eval({X:X_test, Y:Y_test})
            test_acc.append(acc)
            print("Accuracy:", acc)
            
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
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X:X_test, Y:Y_test}))