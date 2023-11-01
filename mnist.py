################################################################################
##                                                                            ##
##                               File: mnist.py                               ##
##                                                                            ##
##            Description: Pre-processing, splitting, displaying,             ##
##                 and general information on MNIST 8x8 data.                 ##
##                                                                            ##
##                Authors: Ali Alper Atasoglu, Thibault Clara                 ##
##                                                                            ##
################################################################################

########################################
#                                      #
#               PACKAGES               #
#                                      #
########################################

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

'''
# These are your training samples along with their labels
mnist_8x8_train = np.load("MNIST/mnist_train.npy")
mnist_8x8_labels = np.load("MNIST/mnist_train_labels.npy")

# These are unknown instances that you should classifya
mnist_unknown = np.load("MNIST/mnist_unknown.npy")
'''

########################################
#                                      #
#                CLASS                 #
#                                      #
########################################

class MNIST:
    '''Class for Accessing the Data from the MNSIT Dataset'''
    def __init__(self, X, y, random_state: int = 42) -> None:
        '''Initializer setting the whole feature space X and corresponding labels y'''
        self.X = X
        self.y = y
        self.random_state = random_state
        # Common Attributes used for Comparison (pasted here because of DummyClassifier):
        self.accuracy_train = None
        self.accuracy_val = None
        self.training_time = None
        self.inference_time = None
    
    def split(self, test_size: float, cv_size: float | None = None) -> None:
        '''Create a training/test split. Alternatively, you can have a validation set'''
        if cv_size is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                      test_size=test_size, random_state=self.random_state)
        else:
            self.X_subset, self.X_test, self.y_subset, self.y_test = train_test_split(self.X, self.y,
                                                      test_size=test_size, random_state=self.random_state)
            
            #Calcultating CV subsize
            sub_size = cv_size / (1 - test_size) 
            self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(self.X_subset, self.y_subset,
                                                test_size=sub_size, random_state=self.random_state)

    def preprocessing(self, nxn_count: int = 8) -> None:
        '''Preprocess the data by reshaping arrays, scaling to [0, 1] and centering the images'''
        # Change 64 to the area of the n x n sized pixel images
        X_train, y_train, X_test, y_test = np.array(self.X_train), np.array(self.y_train), \
            np.array(self.X_test), np.array(self.y_test)
        
        X_reshaped = np.empty((self.X.shape[0], nxn_count ** 2))
        X_train_reshaped = np.empty((X_train.shape[0], nxn_count ** 2))
        X_test_reshaped = np.empty((X_test.shape[0], nxn_count ** 2))

        #Also somehow center the images - possibly here before reshape

        for i in range(self.X.shape[0]):
            flattened_features = self.X[i, :, :].ravel()
            max_val = np.amax(flattened_features)
            X_reshaped[i] = flattened_features / max_val

        for i in range(X_train.shape[0]):
            flattened_features = X_train[i, :, :].ravel()
            max_val = np.amax(flattened_features)
            X_train_reshaped[i] = flattened_features / max_val

        for i in range(X_test_reshaped.shape[0]):
            flattened_features = X_test[i, :, :].ravel()
            max_val = np.amax(flattened_features)
            X_test_reshaped[i] = flattened_features / max_val
            
        if hasattr(self, 'X_cv'):
            X_cv, y_cv = np.array(self.X_cv), np.array(self.y_cv)

            X_cv_reshaped = np.empty((X_cv.shape[0], nxn_count ** 2))
            for i in range(X_cv_reshaped.shape[0]):
                flattened_features = X_cv[i, :, :].ravel()
                max_val = np.amax(flattened_features)
                X_cv_reshaped[i] = flattened_features / max_val 
            
        #Result
        self.X = X_reshaped
        self.X_train = X_train_reshaped
        self.X_test = X_test_reshaped

    def display(self, data, num_images_displayed: int = 50) -> None:
        fig, axes = plt.subplots(num_images_displayed // 10, num_images_displayed % 10)

        for i in range(num_images_displayed):
            row, col = i // 10, i % 10
            ax = axes[row, col]
            ax.imshow(data, cmap='gray')
            ax.axis('off')
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
    
    def data_information(self) -> None:
        classes = sorted(list({label for label in self.y}))
        classes_count = len(classes)
        count = {digit: self.y.count(digit) for digit in classes}
        print(f'There are {classes_count} classes of the data and these classes are {classes}')
        print(f'Every class has the following instace distribution {count}')