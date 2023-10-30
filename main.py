#######################################################################################################
##                                                                                                   ##
##                                           File: main.py                                           ##
##                                                                                                   ##
##                                           Description:                                            ##
##  Used to execute all datasets and models for a comparative study on the most-performing models.   ##
##     All functions are called externally to pre-defined classes and associated sub-functions.      ##
##Please refer to these files to understand exactly how the optimization and training is carried out.##
##                                                                                                   ##
##                            Authors: Ali Alper Atasoglu, Thibault Clara                            ##
##                                                                                                   ##
#######################################################################################################

########################################
#                                      #
#               PACKAGES               #
#                                      #
########################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss, accuracy_score
import numpy as np

########################################
#                                      #
#                CLASSES               #
#                                      #
########################################

import mnist, ninefour
from models import KNeighborsClassification, Data_PCA, DecisionTreeClassification, SVCClassification, SGDClassification

########################################
#                                      #
#               MNIST 8X8              #
#                                      #
########################################

'''
MNIST_8x8 = mnist.MNIST(np.load("data/mnist_train.npy"), np.load("data/mnist_labels.npy"), random_state=42)
MNIST_8x8.split(test_size=0.20)
MNIST_8x8.preprocessing(nxn_count=8)


KNN_MNIST_8x8 = KNeighborsClassification(X=MNIST_8x8.X, y=MNIST_8x8.y, 
                                         X_train=MNIST_8x8.X_train, y_train=MNIST_8x8.y_train, 
                                         X_test=MNIST_8x8.X_test, y_test=MNIST_8x8.y_test)
KNN_MNIST_8x8.train
KNN_MNIST_8x8.CV_tune_K
'''

########################################
#                                      #
#            US CENSUS DATA            #
#                                      #
########################################

us = ninefour.USdata()
us.data_information(False)
#col = us.get_columns()
#print(col)
#us.dummy(True)
#us.display()
#us.unique()
us.split(test_size=0.4, cv_size=0.2)
us.preprocess(with_mean=False)
us.remove_nan()
#us.unique()
us.tonumpy()

'''
# These are your training samples along with their labels
X = mnist_8x8_train = np.load("data/mnist_train.npy")
y = mnist_8x8_labels = np.load("data/mnist_train_labels.npy")

# These are unknown instances that you should classifya
mnist_unknown = np.load("data/mnist_unknown.npy")

mn = mnist.MNIST(X, y)
mn.split(test_size=0.15,cv_size=0.15)
mn.preprocessing()
'''


### K NEIGHBORS ###

#KNN_US = KNeighborsClassification(X=us.X, y=us.y, 
                                         #X_train=us.X_train, y_train=us.y_train, 
                                         #X_test=us.X_test, y_test=us.y_test)
#KNN_US.CV_tune_K(max_n_neighbors=20)
#KNN_US.train()

### PRINCIPAL COMPONENT ANALYSIS ###

#pcamodel = Data_PCA(us, threshold=0.99999999999)
#pcamodel.plot_components()

#pcamodel.change_data(n_components=3)
#us.X = pcamodel.X
#print(us.X)
#us.X_cv

### SUPPORT VECTOR MACHINE ###

#svm = SVCClassification(us.X, us.y, us.X_train, us.y_train, us.X_test, us.y_test, us.X_cv, us.y_cv)
#svm.optimize()
#svm.train(us.X_train, us.y_train, us.X_test, us.y_test)

### DECISION TREE ###

#dt = DecisionTreeClassification()
#dt.optimize(us.X_train, us.y_train, us.X_test, us.y_test, max_max_depth =  5, max_min_samples_leaf =  5)
#dt.optimize(MNIST_8x8.X_train, MNIST_8x8.y_train, MNIST_8x8.X_test, MNIST_8x8.y_test, max_max_depth =  5, max_min_samples_leaf =  5)
#dt.train(us.X_train, us.y_train, col)

sgd = SGDClassification(us.X, us.y, us.X_train, us.y_train, us.X_test, us.y_test, us.X_cv, us.y_cv)
#sgd.optimize()
sgd.train(show_loss=False, show_acc=True)

#svc = SVCClassification(us.X, us.y, us.X_train, us.y_train, us.X_test, us.y_test, us.X_cv, us.y_cv)
#svc.optimize()
#svc.train(us.X_train, us.y_train, us.X_test, us.y_test)

class Compare:
    def __init__(self, dummy_model, KNN_model, SVC_model, DT_model, SGD_model) -> None:
        self.dummy_model = dummy_model
        self.KNN_model = KNN_model
        self.SVC_model = SVC_model
        self.DT_model = DT_model
        self.SGD_model = SGD_model

    def tabulate(self):
        comparison_table = BeautifulTable()
        comparison_table.columns.header = ["Algorithm", "Test Accuracy", "Average Training Time", "Average Inference Time"]
        comparison_table.rows.append(
            ['Dummy (Baseline)', self.dummy_model.accuracy_test, self.dummy_model.training_time, self.dummy_model.inference_time],
            ['KNN', self.KNN_model.accuracy_test, self.KNN_model.training_time, self.KNN_model.inference_time],
            ['SVC', self.SVC_model.accuracy_test, self.SVC_model.training_time, self.SVC_model.inference_time],
            ['DT', self.DT_model.accuracy_test, self.DT_model.training_time, self.DT_model.inference_time],
            ['SGD', self.SGD_model.accuracy_test, self.DT_model.training_time, self.DT_model.inference_time]                                     
                                     )
        print(comparison_table)


    def bar_charts(self):
        # Plot all column space comparatively. 
        pass

#compare = Compare(KNN_model=, SVC_model=, DT_model=, SGD_model=)