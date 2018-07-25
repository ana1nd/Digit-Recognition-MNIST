# MNIST Digit Recognition * 

The data has been downloaded from [here](https://pjreddie.com/projects/mnist-in-csv/). Training data consists of 60k digits each with its own digit label. While testinng data consists of 10k digits. Task is to predict the label  for test dataset.

* Training data:
  * Number of examples: 60K
* Testing data:
  * Number of examples: 10K

* ### Single Layer Perceptron Model

**The parameters setting and vatious architectural details for the 1st convolution model is as follows**
1. Number of hidden layers: 1
2. Number of nodes in hidden layer:256

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
  
  * ### Second approach will be to try out convolution neural network approach
  #### Type- 1 CNN
  
  > For the 1st CNN model, only one layer of convolution followed by RelU activation function and max pooling 
  is applied. Finally the last layer is a fully connected layer which is then fully connected to output layer. 
  The output layer contains 10 nodes, since there can be 10 possible digits from 0 to 9.
  
  **The parameters setting and vatious architectural details for the 1st convolution model is as follows**
  
  1. Number of Convolution filter layer: 1
  2. Number of distinct filters in each layer
      1. 1st layer = 32
  3. Activation function applied on each pixel after convolution: RelU function: R(x) = max(f(x),0)
  4. Type of pooing used: Max Pooling
  5. Number of nodes in fully connected layer: 256
  6. Type of training used: Batch training where size of each batch is 128.
  
  Various other parameters like number of epochs can be tried out by runnign the exp number of times.
The experiment is run for about 50 epochs. Following is the accuracy in the inital epochs
  
 >Epoch =0, Cost=16954186, Test correct = 8789, Test Accuracy = 87.898790  
 Epoch =1, Cost=3019154  
 Epoch =2, Cost=1640675, Test correct = 9181, Test Accuracy = 91.819182  
 Epoch =3, Cost=1047567   
 Epoch =4, Cost=727062, Test correct = 9286, Test Accuracy = 92.869287  
 Epoch =5, Cost=542538  
 Epoch =6, Cost=401969, Test correct = 9327, Test Accuracy = 93.279328  

While Following is the accuracy in the last epochs
 >Epoch =30, Cost=18635, Train Accuracy = 98.804000, Test correct = 9655, Test Accuracy = 96.559656  
 Epoch =31, Cost=17348  
 Epoch =32, Cost=18810, Train Accuracy = 99.080000, Test correct = 9662, Test Accuracy = 96.629663  
 Epoch =33, Cost=18083  
 Epoch =34, Cost=16470, Train Accuracy = 98.892000, Test correct = 9658, Test Accuracy = 96.589659  
 Epoch =35, Cost=12582  
 Epoch =36, Cost=15092, Train Accuracy = 99.096000, Test correct = 9680, Test Accuracy = 96.80968  

### Type-2 CNN

> For the 2nd CNN model, one layer of convolution followed by RelU activation function and max pooling 
  is applied. The output of max pooling is again fed into another layer of convolution filters on which RelU function is applied. Followed by max pooling on the top of RelU layer. Finally the last layer is a fully connected layer which is then fully connected to output layer. 
  
  
   **The parameters setting and vatious architectural details for the 1st convolution model is as follows**
  
  1. Number of Convolution filter layer: 2
  2. Number of distinct filters in each layer
      1. 1st layer = 32
      2. 2nd layer = 64
  3. Activation function applied on each pixel after convolution: RelU function: R(x) = max(f(x),0)
  4. Type of pooing used: Max Pooling on both levels 1st and 2nd
  5. Number of nodes in fully connected layer: 512
  6. Type of training used: Batch training where size of each batch is 128.
  
  Various other parameters like number of epochs can be tried out by runnign the exp number of times.
The experiment is run for about 50 epochs. Following is the accuracy in the inital epochs

 >Epoch =0 ,Cost=624086942, Test correct = 9097, Test Accuracy = 90.979098  
 Epoch =1 ,Cost=74376897  
 Epoch =2 ,Cost=38633793, Test correct = 9457, Test Accuracy = 94.579458  
 Epoch =3 ,Cost=24287078  
 Epoch =4 ,Cost=17375053, Test correct = 9513, Test Accuracy = 95.139514  
 Epoch =5 ,Cost=12896719  
 Epoch =6 ,Cost=9875350, Test correct = 9573, Test Accuracy = 95.739574  
 Epoch =7 ,Cost=7860223  
 Epoch =8 ,Cost=5971894, Test correct = 9590, Test Accuracy = 95.909591  
 Epoch =9 ,Cost=4954969  
 Epoch =10 ,Cost=4035162, Test correct = 9641, Test Accuracy = 96.419642  

While Following is the accuracy in the last epochs
 >Epoch=40, Cost=327456, Train Accuracy = 99.64, Test correct = 9813, Test Accuracy = 98.13  
 Epoch =41 ,Cost=385546  
 Epoch =42 ,Cost=282355, Train Accuracy = 99.62, Test correct = 9807, Test Accuracy = 98.07  
 Epoch =43 ,Cost=329631   
 Epoch =44 ,Cost=336886, Train Accuracy = 99.74, Test correct = 9853, Test Accuracy = 98.53  
 Epoch =45 ,Cost=285817   
 Epoch =46 ,Cost=255326, Train Accuracy = 99.70, Test correct = 9842, Test Accuracy = 98.42  
 Epoch =47 ,Cost=293700  
 Epoch =48 ,Cost=311157, Train Accuracy = 99.73, Test correct = 9815, Test Accuracy = 98.15  
 Epoch =49 ,Cost=240543  
 Epoch =50 ,Cost=269753, Train Accuracy = 99.71, Test correct = 9844, Test Accuracy = 98.44  

