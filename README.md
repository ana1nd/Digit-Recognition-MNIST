## MNIST Digit Recognition 

The data has been downloaded from [here](https://pjreddie.com/projects/mnist-in-csv/). Training data consists of 60k digits each with its own digit label. While testinng data consists of 10k digits. Task is to predict the label  for test dataset.

* First approach is a simple multilayer perceptron model with 1 hidden layer of size 256.  
Various settings of parameters like **learning rate**, **epochs** have been tried out.  
The final setting has been learning rate=0.001 and epochs=100  

The last **10 epochs** have been shown below

  >Epoch = 91,train correct = 57089.00, train accuracy = 95.15,test correct =9480.00, test accuracy = 94.81%
  Epoch = 92,train correct = 57057.00, train accuracy = 95.10,test correct =9506.00, test accuracy = 95.07%
  Epoch = 93,train correct = 56854.00, train accuracy = 94.76,test correct =9440.00, test accuracy = 94.41%
  Epoch = 94,train correct = 56699.00, train accuracy = 94.50,test correct =9449.00, test accuracy = 94.50%
  Epoch = 95,train correct = 56866.00, train accuracy = 94.78,test correct =9448.00, test accuracy = 94.49%
  Epoch = 96,train correct = 56992.00, train accuracy = 94.99,test correct =9482.00, test accuracy = 94.83%
  Epoch = 97,train correct = 56972.00, train accuracy = 94.95,test correct =9474.00, test accuracy = 94.75%
  Epoch = 98,train correct = 57011.00, train accuracy = 95.02,test correct =9487.00, test accuracy = 94.88%
  Epoch = 99,train correct = 57116.00, train accuracy = 95.19,test correct =9484.00, test accuracy = 94.85%
  Epoch = 100,train correct = 57172.00, train accuracy = 95.29,test correct =9503.00, test accuracy = 95.04%
  
  * Second approach will be to try out convolution neural network approach

