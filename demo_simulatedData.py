##########################################
# Demo file demonstrating logistic regression
# using fastgradientDescent on a simple
# simulated dataset
##########################################

import matplotlib.pyplot as mpl
import numpy as np
import logRegFastGradientDescent
from sklearn import preprocessing

# Creating a dummy dataset
features1 = np.append( np.repeat( 100, 30), np.repeat( 500, 30 ) )
features2 = np.append( np.repeat( 20, 30), np.repeat( 40, 30 ) )
features = np.column_stack((features1, features2))
features = preprocessing.scale(features)
labels = np.append( np.repeat( 1, 30), np.repeat( -1,30 ) )
lamda = 1
maxIter = 1000
InitStepSize = 0.01

# Running Fast Gradient Descent
print("Running Logistic Regression with Fast Gradient Descent..")
( lossFunctionValuesFastGrad, betaValuesFastGrad ) = logRegFastGradientDescent.fastGradDescent( features, labels, lamda, InitStepSize, maxIter )

# Find Error on training set
finalBeta = betaValuesFastGrad[-1]
print("Error on Training set = ",  logRegFastGradientDescent.misclassificationError( finalBeta, features, labels ) )

# Plotting loss function with iterations
mpl.plot( range(maxIter), lossFunctionValuesFastGrad )
mpl.title("Loss Function values with iterations")
mpl.show()



