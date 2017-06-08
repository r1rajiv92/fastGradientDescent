##########################################
# Demo file demonstrating logistic regression
# using fastgradientDescent on a real world dataset
##########################################

import pandas as pd
import matplotlib.pyplot as mpl
import logRegFastGradientDescent
from sklearn import preprocessing

#######################
# Here, you can find the spam.data.txt in the repository
# This data was obtained from https://statweb.stanford.edu/~tibs/ElemStatLearn/
#######################

data = pd.read_csv('spam.data.txt', header = None, sep = ' ')
features = data[data.columns[0:57]]
labels = data[57]
labels[ labels == 0 ] = -1

# Splitting dataset into train and test set (80:20)
print("Splitting the spam dataset into train and test...")
featuresTrain = features[ 0 : int( 0.8 * len( features ) ) ]
featuresTest = features[ int( 0.8 * len( features ) ) + 1: ]
labelsTrain = labels[ 0 : int( 0.8 * len( labels ) ) ]
labelsTest = labels[ int ( 0.8 * len( labels ) ) + 1: ]

# Scaling the dataset
print("Scaling the dataset...")
scaler = preprocessing.StandardScaler().fit(featuresTrain)
featuresTrain = scaler.transform(featuresTrain)
featuresTest = scaler.transform(featuresTest)

lamda = 1
maxIter = 1000
InitStepSize = 0.01

print("Running Logistic Regression with Fast Gradient Descent..")
# Running Fast Gradient Descent
( lossFunctionValuesFastGrad, betaValuesFastGrad ) = logRegFastGradientDescent.fastGradDescent( featuresTrain, labelsTrain, 0.01, InitStepSize, maxIter )

# Find Error on training set
finalBeta = betaValuesFastGrad[-1]
print("Error on Training set = ",  logRegFastGradientDescent.misclassificationError( finalBeta, featuresTrain, labelsTrain ) )

# Find Error on test set
print("Error on Test set = ",  logRegFastGradientDescent.misclassificationError( finalBeta, featuresTest, labelsTest ) )

# Plotting loss function with iterations
mpl.plot( range(maxIter), lossFunctionValuesFastGrad )
mpl.title("Loss Function values with iterations")
mpl.show()




