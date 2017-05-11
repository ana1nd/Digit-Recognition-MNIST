# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:38:05 2017

@author: anand
"""
from __future__ import division
import os
import numpy as np
import pandas as pd
import tensorflow as tf

keep_rate = 0.8
batch_size = 128
keep_prob = tf.placeholder(tf.float32)

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
    
def CNN(X):
    x = tf.reshape(X,shape=[-1,28,28,1])
    
    W1 = tf.Variable(tf.random_normal([5,5,1,32])) #32 distinct filters
    b1 = tf.Variable(tf.random_normal([32])) #32 biases for each filters
    
    #convolve X with W1 & b1
    conv1 = tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='SAME')    
    conv1 = tf.nn.relu(conv1+b1)
    
    # apply max pool filter
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    W2 = tf.Variable(tf.random_normal([5,5,32,64])) #32 distinct filters
    b2 = tf.Variable(tf.random_normal([64])) #32 biases for each filters
    
    #convolve X with W1 & b1
    conv2 = tf.nn.conv2d(conv1,W2,strides=[1,1,1,1],padding='SAME')    
    conv2 = tf.nn.relu(conv2+b2)
    
    # apply max pool filter
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    #matrix mul with fully connnected layer
    W_fc = tf.Variable(tf.random_normal([7*7*64,512]))    
    b_fc = tf.Variable(tf.random_normal([512]))
    
    fc = tf.reshape(conv2,[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, W_fc)+b_fc)
    fc = tf.nn.dropout(fc, keep_rate)
    
    W_last = tf.Variable(tf.random_normal([512,10]))
    b_last = tf.Variable(tf.random_normal([10]))
    output = tf.matmul(fc, W_last)+ b_last
    
    #output = tf.arg_max(output,1)    
    return output

    
def main():
    print("Begin")
    train_X, test_X, train_y, test_y = get_mnist_data()
    
    train_X = np.delete(train_X,0,axis=1)    #deleting bias column
    test_X = np.delete(test_X,0,axis=1)
    
    x_size = train_X.shape[1]   # Number of input nodes: 784 features and 1 bias
    #h_size = 256 #256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes 10 possible outcomes

    
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    
    print("CNN Begin")
    predict = CNN(X)
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))
    #updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(predict,y) )
    updates = tf.train.AdamOptimizer().minimize(cost)
    
    print("Begin Session")
    sess = tf.Session()
    init = tf.initialize_all_variables()
    #init = tf.global_variables_initializer()
    sess.run(init)
    
    print("Begin Epochs")
    for epoch in range(40,60,1):
        loss = 0
        num_of_batches = int(len(train_X)/batch_size)
        for i in range(1,num_of_batches+1,1):
            next_x = train_X[batch_size*i:min((i+1)*batch_size,len(train_X)),:]
            next_y = train_y[batch_size*i:min((i+1)*batch_size,len(train_y)),:]
            c = sess.run([cost,updates], feed_dict={X: next_x, y: next_y})
            loss += c[0]
        print("Epoch =%d ,Cost=%d "%(epoch,loss))
        
        if(epoch%2==0):
            print("Epoch=%d"%(epoch))
            
            #TRAIN
            low = 0
            high = 10000
            var_train = sess.run(predict,feed_dict={X: train_X[low:high], y: train_y[low:high]})
            var_train = np.argmax(var_train,axis=1)
            train_accuracy = np.mean(np.argmax(train_y[low:high], axis=1) == var_train)
            train_correct = np.count_nonzero(np.argmax(train_y[low:high], axis=1) == var_train)#/float(len(train_y))
            print("Train correct = %d, Train Accuracy = %f" %(train_correct,train_accuracy*100))    
        
        
            #TEST    
            var_test = sess.run(predict,feed_dict={X:test_X,y:test_y})
            var_test = np.argmax(var_test,axis=1)
            test_accuracy = np.mean(np.argmax(test_y, axis=1) == var_test)
            test_correct = np.count_nonzero(np.argmax(test_y, axis=1) == var_test)#/float(len(test_y))
            print("Test correct = %d, Test Accuracy = %f" %(test_correct,test_accuracy*100))    
    
    sess.close()
    print("Done")
    

if __name__ == '__main__':
    main()
    
    
    
    
#import tensorflow as tf
#X = tf.placeholder("float", shape=[None, x_size])
#predict = CNN(X)
#init = tf.initialize_all_variables()
#sess = tf.Session()
#sess.run(init)
#
#array = predict.eval(sess)
#print (array)