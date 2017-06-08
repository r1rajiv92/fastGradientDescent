import pandas as pd
import numpy as np


################
# Function that computes gradient of loss function
# beta - coefficients in the logistic regression functions
# x - features 
# y - labels
# lamda - lamda value for l2 regularization
#################
def computeGrad( beta, x, y, lamda ):
    
    temp = 1 / ( 1 + np.exp( y * x.dot( beta ) ) )
    deltaFunction = np.transpose( x ).dot( -1 * y * temp )
    deltaFunction = deltaFunction / len( x ) + 2 * lamda * beta
    return deltaFunction


################
# Function that computes the loss value
# beta - coefficients in the logistic regression functions
# x - features 
# y - labels
# lamda - lamda value for l2 regularization
#################
def computeLossFunction( beta, x, y, lamda ):

    temp = np.exp( np.multiply( y * x.dot( beta ), -1 ) )
    temp = sum( np.log( 1 + temp ) ) / len( x )
    loss = temp + sum( lamda * beta * beta )
    return loss


################
# Function that calculates the misclassfication error
# beta - coefficients in the logistic regression functions
# x - features 
# y - labels
#################
def misclassificationError( beta, x, y ):

    prediction = 1 / ( 1 + np.exp( -1 * x.dot( beta ) ) )
    prediction = prediction > 0.5
    y[ y == -1 ] = 0
    errorRate =  sum( prediction != y ) / len( y )

    return errorRate


################
# Function that determines if backtracking is needed
# beta - coefficients in the logistic regression functions
# x - features 
# y - labels
# lamda - lamda value for l2 regularization
# alpha - parameters for backtracking
#################
def calcBackTrackingCondition( beta, x, y, lamda, alpha, Eta ):
    
    left = computeLossFunction( beta - Eta*computeGrad( beta, x, y, lamda ), x, y, lamda )
    right = computeLossFunction( beta, x, y, lamda ) + alpha * Eta * sum( np.power( computeGrad( beta, x, y, lamda ), 2 ) )
    if left == float( 'Inf' ) or right == float( 'Inf' ):
        return 0
    if left <= right:
        return 1
    else:
        return 0


################
# Function that runs backtracking to calculate optimal step size
# prevEta - step size in the previous iteration
# x - features 
# y - labels
# lamda - lamda value for l2 regularization
# beta - coefficients in the logistic regression functions
# alpha, gamma  - parameters for backtracking
#################
def backTracking ( prevEta, x, y, lamda, beta, alpha, gamma ):

    Eta = prevEta
    while( calcBackTrackingCondition( beta, x, y, lamda, alpha, Eta ) == 0 ):
        Eta = gamma*Eta

    return Eta


################
# Function that runs fast gradient descent
# x - features 
# y - labels
# lamda - lamda value for l2 regularization
# stepSizeInit - Initial Step size for fastGradDescent
# maxIter - maximum number of iterations for fastGradDescent
#################
def fastGradDescent( x, y, lamda, stepSizeInit, maxIter ):

    beta = pd.Series( [0] * (  x.shape[1] ) )
    theta = pd.Series( [0] * ( x.shape[1] ) )
    stepSize = stepSizeInit
    lossFunctionValues = []
    betaValues = []
    for i in range( maxIter ):
        betaValues.append( beta ) 
        lossFunctionValues.append( computeLossFunction( beta, x, y, lamda ) )
        stepSize = backTracking( stepSize, x, y, lamda, beta, 0.5, 0.8 )

        betaPrev = beta
        beta = theta - stepSize * computeGrad( beta, x, y, lamda )
        theta = beta + i * ( beta - betaPrev )/( i+3 )

    return lossFunctionValues, betaValues
