import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1+np.exp(-z))

def sigmoidPrime(z):
    return (np.exp(-z)/(1+np.exp(-z))**2)

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Things to do for preprocessing step:
     - remove features that have the same value for all data points
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - divide the original data set to training, validation and testing set"""

    mat = loadmat('mnist_sample.mat') #loads the MAT object as a Dictionary
    n_valid = 5000
    train_data = np.concatenate((mat['train0'], mat['train1'],
                                 mat['train2'], mat['train3'],
                                 mat['train4'], mat['train5'],
                                 mat['train6'], mat['train7'],
                                 mat['train8'], mat['train9']), 0)
    train_label = np.concatenate((np.ones((mat['train0'].shape[0], 1), dtype='uint8'),
                                  2 * np.ones((mat['train1'].shape[0], 1), dtype='uint8'),
                                  3 * np.ones((mat['train2'].shape[0], 1), dtype='uint8'),
                                  4 * np.ones((mat['train3'].shape[0], 1), dtype='uint8'),
                                  5 * np.ones((mat['train4'].shape[0], 1), dtype='uint8'),
                                  6 * np.ones((mat['train5'].shape[0], 1), dtype='uint8'),
                                  7 * np.ones((mat['train6'].shape[0], 1), dtype='uint8'),
                                  8 * np.ones((mat['train7'].shape[0], 1), dtype='uint8'),
                                  9 * np.ones((mat['train8'].shape[0], 1), dtype='uint8'),
                                  10 * np.ones((mat['train9'].shape[0], 1), dtype='uint8')), 0)
    test_label = np.concatenate((np.ones((mat['test0'].shape[0], 1), dtype='uint8'),
                                 2 * np.ones((mat['test1'].shape[0], 1), dtype='uint8'),
                                 3 * np.ones((mat['test2'].shape[0], 1), dtype='uint8'),
                                 4 * np.ones((mat['test3'].shape[0], 1), dtype='uint8'),
                                 5 * np.ones((mat['test4'].shape[0], 1), dtype='uint8'),
                                 6 * np.ones((mat['test5'].shape[0], 1), dtype='uint8'),
                                 7 * np.ones((mat['test6'].shape[0], 1), dtype='uint8'),
                                 8 * np.ones((mat['test7'].shape[0], 1), dtype='uint8'),
                                 9 * np.ones((mat['test8'].shape[0], 1), dtype='uint8'),
                                 10 * np.ones((mat['test9'].shape[0], 1), dtype='uint8')), 0)
    test_data = np.concatenate((mat['test0'], mat['test1'],
                                mat['test2'], mat['test3'],
                                mat['test4'], mat['test5'],
                                mat['test6'], mat['test7'],
                                mat['test8'], mat['test9']), 0)


   # remove features that have same value for all points in the training data
    same = [True] * 784
    print (test_data.shape)
    print (train_data.shape)


    for j in range(len(train_data)-1):
        for x in range(0, len(train_data[0])-1):
            if train_data[j].item(x) != train_data[j+1].item(x):
                same[x] = False
    # remove common features
    n = 0
    while n < (len(train_data[0])-1):
        if same[n]:
            train_data = np.delete(train_data,n,1)
        n += 1

    # remove common features
    n = 0
    while n < (len(test_data[0])-1):
        if same[n]:
            test_data = np.delete(test_data,n,1)
        n += 1

    # convert data to double
    train_data = np.double(train_data)
    test_data = np.double(test_data)
    # normalize data to [0,1]
    train_data = train_data / 255
    test_data = test_data / 255

    # Split train_data and train_label into train_data, validation_data and train_label, validation_label
    # replace the next two lines
    r = np.random.permutation(range(0, train_data.shape[0]))
    size = len(r)

    #percent of train data to be split
    tr = .9
    va = 1-tr

    training = int(size*tr)
    val = int(size*va)

    validation_data = np.array([])
    validation_label = np.array([])

    validation_data = train_data[r[val:],:]
    validation_label = train_label[r[val:],:]

    train_data = train_data[r[0:training],:]
    train_label = train_label[r[0:training],:]

    print("preprocess done!")

    return train_data, train_label, validation_data, validation_label, test_data, test_label


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
    #start gradient for w2
    truth_label = np.zeros((afterTest.shape[0], afterTest.shape[1]))

    #determine the error of the weights associated with the output layer
    for x in range(0,n):
        truth_label[x, int(training_label[x])-1] = 1

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

    print (front.shape)
    print (testInitial.shape)
    print (w1.shape)
    #Ojconcat = np.c_[Ojconcat,ones]
    print ("ADAM")
    #w1 = np.c_[np.transpose(w1),np.ones(w1.shape[1])]
    #w1 = np.transpose(w1)
    print (w1.shape)
    #calculate function 10
    Jw1 = np.transpose(temp).dot(testInitial)

    #get the lam value to go inside 17
    lam1 = lambdaval*w1

    #grad_w1 based on 17
    grad_w1 = (np.add(lam1,Jw1))/n

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    JW12Sum = np.sum((truth_label * np.log(afterTest)) + ((1 - truth_label) * np.log(1-afterTest)))
    JW12 = (-1/n) * JW12Sum

    w1Sum = w1.sum()
    w2Sum = w2.sum()
    obj_val = JW12 + lambdaval/(2*n) * (w1Sum**2 + w2Sum**2)


    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""
    #add a column of ones to training data for the bias nodes
    n = data.shape[0]
    ones = [1]*n
    print ("FLAG")
    print (data.shape)
    print (w1.shape)

    #take data and apply w1 to it
    w1concat = np.c_[data, ones]
    presigw1 = w1concat.dot(np.transpose(w1))

    #apply sigmoid
    postsigw1 = sigmoid(presigw1)

    ones = [1]*postsigw1.shape[0]

    #take data and apply w2 to it
    w2concat = np.c_[presigw1,ones]
    presigw2 = w2concat.dot(np.transpose(w2))

    #apply sigmoid
    postsigw2 = sigmoid(presigw2)

    #put lables on each function
    labels = np.empty([data.shape[0], 1])

    for index in range(0,postsigw2.shape[0]):
        labels[index] = np.argmax(postsigw2[index])+1

    #labels = np.amax(test, axis = 0)
    #print (labels)
    return labels

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
