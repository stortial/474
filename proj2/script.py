import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A k x d matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # print(X)
    # print(y)
    d = X.shape[1]
    numK = []
    #find k
    for x in range(y.shape[0]):
        if int(y[x]) not in numK:
            numK.append(int(y[x]))

    k = len(numK)

    # Pre build the matricies
    Xc = np.ones(X.shape)
    means = np.zeros((k,d))
    total = np.zeros((k,1))

    #find mean
    for index in range(y.shape[0]):
        #increment totals at position
        total[int(y[index])-1]+=1
        #increment for each d
        for dIter in range(d):
            means[int(y[index])-1][dIter] += X[index][dIter]


    #divide by d to find the means
    for row in range(k):
        for column in range(d):
            means[row][column] = means[row][column]/total[row]



    # Pre build the matricies
    #means = np.ones(X.shape[1])
    Xc = np.ones(X.shape)

    # Find means, the average values of each column of X
    # Then, find Xc, an intermediate term for finding covmat
    for cols in range(X.shape[1]):
        temp = X[:,cols]
        print(temp.shape)
        colAvg = np.average(temp)
        Xc[:,cols] = temp - colAvg
        #means[cols] = colAvg
    covmat = (1/X.shape[0])*(w.transpose(Xc).dot(Xc))
    print(means)

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example


    d = X.shape[1]

    numK = []
    #find k
    for x in range(y.shape[0]):
        if int(y[x]) not in numK:
            numK.append(int(y[x]))

    k = len(numK)

    # Pre build the matricies
    Xc = np.ones(X.shape)
    means = np.zeros((k,d))
    total = np.zeros((k,1))

    #find mean
    for index in range(y.shape[0]):
        #increment totals at position
        total[int(y[index])-1]+=1
        #increment for each d
        for dIter in range(d):
            means[int(y[index])-1][dIter] += X[index][dIter]


    #divide by d to find the means
    for row in range(k):
        for column in range(d):
            means[row][column] = means[row][column]/total[row]


    #covariance--------------------------

    covmats = []

    classes = []

    #initialize matricies
    #these are just to split up the data so we can call bens thing to get covariance

    for x in range(k):
        classes.append(np.zeros((1,d)))

    for row in range(y.shape[0]):
        classes[int(y[row])-1] = np.vstack([classes[int(y[row])-1],X[row]])

    for i in range(k):
        classes[i] = classes[i][1:]


    for index in range(k):
        #prebuld the matrix to help with covariance
        Xc = np.ones(classes[index].shape)

        #find Xc, an intermediate term for finding covmat
        for cols in range(classes[index].shape[1]):
            temp = classes[index][:,cols]
            colAvg = np.average(temp)
            Xc[:,cols] = temp - colAvg
        covmats.append((1/classes[index].shape[0])*(np.transpose(Xc)).dot(Xc))

    # Outputs
    # means - A k x d matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # Matrix of likelihoods of eatch feature for each label
    np.linalg.det(covmat)


    likelihood = np.square(Xtest - means)
    ypred = np.vectorize(ypredHelper)
    print(ypred.shape)

    print(likelihood)
    return acc,ypred

def ypredHelper(a,b):
    if(a>b): return 1
    else: return 0

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    s = np.dot(np.transpose(X),X)

    inverse = np.linalg.inv(s)
    w = np.dot(inverse,np.dot(np.transpose(X),y))
    # IMPLEMENT THIS METHOD
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    D = X.shape[0]

    left = np.linalg.inv(D*lambd*np.identity(X.shape[1]) + np.dot(np.transpose(X),X))
    right = np.dot(np.transpose(X),y)
    w = np.dot(left,right)

    # IMPLEMENT THIS METHOD
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    N = Xtest.shape[0]

    total = np.sum(np.square(np.transpose(ytest-np.dot(Xtest,w))))

    mse = total/N

    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    #changes y from (252,1) to (252,)
    y = y.transpose()
    y = y[0]
    y.tolist()

    N = X.shape[0]

    preSum = y - np.dot(X,w)
    postSum = np.sum(np.dot(np.transpose(preSum),preSum))/(2*N)
    regression = (lambd/2)*(np.dot(np.transpose(w),w))
    error = postSum+regression


    print("HI ADAM")
    print("Error: ", error)

    # IMPLEMENT THIS METHOD
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))

    N = x.shape[0]

    Xp = np.zeros((N,p+1))

    for i in range(N):
        for j in range(p+1):
            Xp[i][j] = x[i]**j

    return Xp

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')
"""
# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
"""

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
"""
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()
"""

# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
