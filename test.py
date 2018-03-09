import numpy as np

# Paste your sigmoid function here
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1+np.exp(-z))

# Paste your nnObjFunction here
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    print(n_input)
    print(n_hidden)
    print(n_class)
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    obj_grad = np.array([])

    # Your code here

    #add a column of ones to training data for the bias nodes


    n = training_data.shape[0]

    ones = [1]*n

    #take data and apply w1 to it
    testInitial = np.c_[training_data, ones]
    test = testInitial.dot(np.transpose(w1))

    #apply sigmoid
    Oj = sigmoid(test)

    #take data and apply w2 to it
    Ojconcat= Oj
   # print(OjWeight.shape[0],OjWeight.shape[1])
    Ojconcat = np.c_[Ojconcat,ones]
   # print(OjWeight.shape[0],OjWeight.shape[1])
    OjWeight = Ojconcat.dot(np.transpose(w2))
    #apply sigmoid
    afterTest = sigmoid(OjWeight)
    #np.set_printoptions(threshold=np.nan)
    print ("HELOOOO")
    print (afterTest.shape)
    #start gradient for w2
    truth_label = np.zeros((afterTest.shape[0], afterTest.shape[1]))

    #determine the error of the weights associated with the output layer
    for x in range(0,n):
        truth_label[x, int(training_label[x])] = 1

    # deltaL = ol - yl
    deltaL =  (truth_label - afterTest)

    # (9)
    Jw2 = -(np.transpose(deltaL).dot(Ojconcat))

    #get the lam value to go inside 16
    lam2 = lambdaval*w2

    #grad_w2 based on 16
    grad_w2 = (np.add(lam2,Jw2))/n

    adam = deltaL.dot(w2)

    #the front of function 12
    front = -(1-Ojconcat)*Ojconcat*adam

    temp = front
    temp = np.transpose(temp)[0:n_hidden]
    temp = np.transpose(temp)

    #calculate function 10
    Jw1 = np.transpose(temp).dot(testInitial)

    #get the lam value to go inside 17
    lam1 = lambdaval*w1

    #grad_w1 based on 17
    grad_w1 = (np.add(lam1,Jw1))/n

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    JW12Sum = np.sum((truth_label * np.log(afterTest)) + ((1 - truth_label) * np.log(1-afterTest)))
    JW12 = (-1/n) * JW12Sum

    Jw12 = 0
    w1Sum = w1.sum()
    w2Sum = w2.sum()
    obj_val = Jw12 + lambdaval/(2*n) * (w1Sum**2 + w2Sum**2)

    return (obj_val, obj_grad)


n_input = 5
n_hidden = 3
n_class = 2
print("a1")
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
print("a2")
training_label = np.array([0,1])
print("a3")
lambdaval = 0
print("a4")
params = np.linspace(-5,5, num=26)
print("a5")
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
print("a6")
objval,objgrad = nnObjFunction(params, *args)
print("a7")
print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)
