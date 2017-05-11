# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:47:53 2017

@author: anand
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# To stop potential randomness
#seed = 128
#rng = np.random.RandomState(seed)
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  
    yhat = tf.matmul(h, w_2)  
    return yhat

def get_mnist_data():
    

    root_dir =  os.getcwd()
    data_dir = os.path.join(root_dir,'data')
    sub_dir = os.path.join(root_dir,'sub')
    
    # check for existence
    os.path.exists(root_dir)
    os.path.exists(data_dir)
    os.path.exists(sub_dir)
    
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    train = np.array(train)
    test = np.array(test)
    
    r_train,c_train = np.shape(train)
    r_test,c_test = np.shape(test)
    
    bias_train = np.ones((r_train,1))
    bias_test = np.ones((r_test,1))
    
    train_x = train[:,1:]
    train_y = train[:,0:1]
    
    test_x = test[:,1:]
    test_y = test[:,0:1]
    
    train_x = np.concatenate((bias_train,train_x),axis=1)
    test_x = np.concatenate((bias_test,test_x),axis=1)
    
    
    num_labels = len(np.unique(train_y))
    all_train_Y = np.eye(num_labels)[train_y]  # One liner trick!
    train_y = all_train_Y.reshape((len(train_y),num_labels))

    all_test_Y = np.eye(num_labels)[test_y]  # One liner trick!
    test_y = all_test_Y.reshape((len(test_y),num_labels))
        
    return train_x,test_x,train_y,test_y
    
    
def main():
    train_X, test_X, train_y, test_y = get_mnist_data()

    
    x_size = train_X.shape[1]   # Number of input nodes: 784 features and 1 bias
    h_size = 256 #256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes 10 possible outcomes

    
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.arg_max(yhat,1)
    
    #predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    # Run session
    sess = tf.Session()
    init = tf.initialize_all_variables()
    #init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        var_train = sess.run(predict,feed_dict={X: train_X, y: train_y})
        var_test = sess.run(predict,feed_dict={X:test_X,y:test_y})
        
        train_accuracy = np.mean(np.argmax(train_y, axis=1) == var_train)
        test_accuracy = np.mean(np.argmax(test_y, axis=1) == var_test)
        
        train_correct = np.count_nonzero(np.argmax(train_y, axis=1) == var_train)#/float(len(train_y))
        test_correct = np.count_nonzero(np.argmax(test_y, axis=1) == var_test)#/float(len(test_y))
        print("Epoch = %d,train correct = %.2f, train accuracy = %.2f,test correct =%.2f, test accuracy = %.2f%%"
              % (epoch + 1,train_correct, 100. * train_accuracy,test_correct,100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()
    
    
    
#import tensorflow as tf
#temp = train_y[0]
#temp = tf.pack(temp)
#temp = tf.arg_max(temp,0)
#init = tf.initialize_all_variables()
#with tf.Session() as s:
#    s.run(init)
#    print (s.run(temp))
