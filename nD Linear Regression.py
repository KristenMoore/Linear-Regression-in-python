import numpy as np
from numpy import arange, zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from mpl_toolkits.mplot3d import Axes3D
# ======= load and visualise the dataset =======
data=np.loadtxt('ex1data2.txt',delimiter=',')
print(data[:9,:])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25)]:
    xvals = data[:, 0]
    yvals = data[:, 1]
    zvals = data[:, 2]
    ax.scatter(xvals, yvals, zvals, c=c, marker=m)
 
ax.set_xlabel('House Size')
ax.set_ylabel('# Bedrooms')
ax.set_zlabel('House Price')
plt.show()

# compare orders of magnitude of variables: house size is approx 1000 times the number of bedrooms
# this suggests we should perform feature scaling to make gradient descent converge more quickly

# ======= format & normalize the data =======
x=data[:,[0,1]]
Y=data[:,-1]
m=len(Y)
y=reshape(Y,(m,1))
# perform feature normalization
def featureNormalize(X):
    X_norm=X
    mean=[]  # need to store mean & std to be able to normalize unseen data
    stdev=[]
    for i in range(shape(X)[1]):
         mu=np.mean(X[:,i])
         mean.append(mu)
         sigma=np.std(X[:,i])
         stdev.append(sigma)
         X_norm[:,i]=(X_norm[:,i]-mu)/sigma
    return X_norm
x=featureNormalize(x)
# add column of ones corresponding to x_0=1
X = ones(shape=(m,3))
X[:, 1:3] = x
    
# ======= define the cost function J for linear regression =======
def cost_function(theta, X, y):
        prediction=dot(X,theta)
        J = (1.0/(2*m))*dot(transpose(prediction-y),(prediction-y))               
        return J

    
# ======= Gradient Descent =======
# define function for batch gradient descent
def gradient_descent(theta, X, y, alpha, num_iters):
    m=len(y)
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):
        theta=theta-(alpha/m)*(dot(transpose(X),(dot(X,theta)-y)))
        J_history[i] = cost_function(theta, X, y)
    return theta, J_history

# set parameters for gradient descent
alpha = 0.01
iterations = 400
theta = zeros(shape=(3, 1)) 

#compute and display initial cost
print('Initial cost: '+str(cost_function(theta,X,y)[0][0]))

# perform gradient descent & check learning rate by plotting cost at each iteration
theta, J_history = gradient_descent(theta, X, y, alpha, iterations)
print('Theta found by gradient descent: '+str(theta[0][0])+', '+str(theta[1][0])+', '+str(theta[2][0]))
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()
