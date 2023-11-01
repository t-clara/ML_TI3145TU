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
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable

########################################
#                                      #
#                CLASSES               #
#                                      #
########################################

import mnist, ninefour
import pandas as pd
from models import KNeighborsClassification, Data_PCA, DecisionTreeClassification, SVCClassification, SGDClassification, Dummy

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
X = pd.read_csv('data\census_train.csv', sep=',', header=0)
y = pd.read_csv('data\census_labels.csv', sep=',', header=0)
us = ninefour.USdata(X=X, y=y)
us.data_information(False)
#print(col)
#us.display()
#us.unique()
us.split(test_size=0.15, cv_size=0.15)
'''
pcamodel = Data_PCA(us.X, us.y, us.X_train, us.y_train, us.X_test, us.y_test, us.X_cv, us.y_cv, threshold=0.99)
us.X = pcamodel.X
us.y = pcamodel.y
self.preprocess()
self.remove_nan()
us.X_train = pcamodel.X_train
us.y_train = pcamodel.y_train
us.X_test = pcamodel.X_test
us.y_test = pcamodel.y_test
us.X_cv = pcamodel.X_cv
us.y_cv = pcamodel.y_cv
'''
us.preprocess(with_mean=True)
us.remove_nan()
col = us.get_columns()
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
### DUMMY ###
dummy = Dummy(us)
dummy.train()

### K NEIGHBORS ###

knn = KNeighborsClassification(us)
knn.train()
#print(knn.accuracy_val, knn.training_time, knn.inference_time, knn.accuracy_test)
#knn.optimize(max_n_neighbors=50)
print(knn.accuracy_val, knn.training_time, knn.inference_time)

### SUPPORT VECTOR MACHINE ###

svm = SVCClassification(us)
#svm.optimize()
svm.train()
#svm.optimize(further_optimize=True)

### DECISION TREE ###

dt = DecisionTreeClassification(us)
#dt.optimize(us.X_train, us.y_train, us.X_test, us.y_test, max_max_depth =  5, max_min_samples_leaf =  5)
#dt.optimize(MNIST_8x8.X_train, MNIST_8x8.y_train, MNIST_8x8.X_test, MNIST_8x8.y_test, max_max_depth =  5, max_min_samples_leaf =  5)
dt.train(labels=col)

sgd = SGDClassification(us)
#sgd.optimize()
sgd.train(show_loss=False, show_acc=True)
#sgd.optimize(further_optimize=True)
#sgd.train(show_loss=False, show_acc=True)
#sgd.display()

svc = SVCClassification(us)
#svc.optimize()
svc.train()

class Compare:
    def __init__(self, dummy_model, DT_model, KNN_model, SVC_model, SGD_model) -> None:
        self.dummy_model = dummy_model
        self.DT_model = DT_model
        self.KNN_model = KNN_model
        self.SVC_model = SVC_model
        self.SGD_model = SGD_model

    def tabulate(self):
        comparison_table = BeautifulTable()
        comparison_table.columns.header = ["Algorithm", "Validation Accuracy", "Average Training Time", "Average Inference Time"]
        comparison_table.rows.append(['Dummy (Baseline)', self.dummy_model.accuracy_val, self.dummy_model.training_time, self.dummy_model.inference_time])
        comparison_table.rows.append(['DT', self.DT_model.accuracy_val[0], self.DT_model.training_time[0], self.DT_model.inference_time[0]])
        comparison_table.rows.append(['SGD', self.SGD_model.accuracy_val[0], self.SGD_model.training_time[0], self.SGD_model.inference_time[0]])    
        comparison_table.rows.append(['KNN', self.KNN_model.accuracy_val[0], self.KNN_model.training_time[0], self.KNN_model.inference_time[0]])
        comparison_table.rows.append(['SVC', self.SVC_model.accuracy_val[0], self.SVC_model.training_time[0], self.SVC_model.inference_time[0]])
        print(comparison_table)

    def bar_chart_validation_accuracy(self):
        untuned = [np.round(self.DT_model.accuracy_val[0]*100, 2), np.round(self.SGD_model.accuracy_val[0]*100, 2), np.round(self.KNN_model.accuracy_val[0]*100, 2), np.round(self.SVC_model.accuracy_val[0]*100, 2)]
        tuned = [np.round(self.DT_model.accuracy_val[-1]*100, 2), np.round(self.SGD_model.accuracy_val[-1]*100, 2), np.round(self.KNN_model.accuracy_val[-1]*100, 2), np.round(self.SVC_model.accuracy_val[-1]*100, 2)]
        model_types = ["DT", "SGD", "KNN", "SVC"]
        df = pd.DataFrame({'Untuned': untuned,'Tuned': tuned}, index=model_types)
        ax = df.plot.bar(rot=0, color={"Untuned": "green", "Tuned": "red"})
        for container in ax.containers:
            ax.bar_label(container)
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Model Validation Accuracies (Untuned vs. Tuned)')
        ax.set_ylim(0, 100)
        plt.show()

        '''
        model_types = ("DT", "SGD", "KNN", "SVC")
        ### REQUIRES IMPLEMENTATION OF MODEL ATTRIBUTES IN models.py ###
        model_accuracies = {
            'Untuned': (self.DT_model.accuracy_val[0], self.SGD_model.accuracy_val[0], self.KNN_model.accuracy_val[0], self.SVC_model.accuracy_val[0]),
            'Tuned': (self.DT_model.accuracy_val[0], self.SGD_model.accuracy_val[0], self.KNN_model.accuracy_val[0], self.SVC_model.accuracy_val[0])
        }

        x = np.arange(len(model_types))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in model_accuracies.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=4.5)
            multiplier += 1
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Model Validation Accuracies (Untuned vs. Tuned)')
        ax.set_xticks(x + width, model_types)
        ax.legend(loc='upper left')
        ax.set_ylim(0, 1)
        plt.show()
        '''

    def bar_chart_training_time(self):
        untuned = [np.round(self.DT_model.training_time[0], 2), np.round(self.SGD_model.training_time[0], 2), np.round(self.KNN_model.training_time[0], 2), np.round(self.SVC_model.training_time[0], 2)]
        tuned = [np.round(self.DT_model.training_time[-1], 2), np.round(self.SGD_model.training_time[-1], 2), np.round(self.KNN_model.training_time[-1], 2), np.round(self.SVC_model.training_time[-1], 2)]
        combined = untuned + tuned
        model_types = ["DT", "SGD", "KNN", "SVC"]
        df = pd.DataFrame({'Untuned': untuned,'Tuned': tuned}, index=model_types)
        ax = df.plot.bar(rot=0, color={"Untuned": "green", "Tuned": "red"})
        for container in ax.containers:
            ax.bar_label(container)
        ax.set_ylabel('Training Time [s]')
        ax.set_title('Model Training Times (Untuned vs. Tuned)')
        ax.set_ylim(0, np.ceil(max(combined)*1.2))
        plt.show()

    def bar_chart_inference_time(self):
        untuned = [np.round(self.DT_model.inference_time[0]*1000, 2), np.round(self.SGD_model.inference_time[0]*1000, 2), np.round(self.KNN_model.inference_time[0]*1000, 2), np.round(self.SVC_model.inference_time[0]*1000, 2)]
        tuned = [np.round(self.DT_model.inference_time[-1]*1000, 2), np.round(self.SGD_model.inference_time[-1]*1000, 2), np.round(self.KNN_model.inference_time[-1]*1000, 2), np.round(self.SVC_model.inference_time[-1]*1000, 2)]
        combined = untuned + tuned
        model_types = ["DT", "SGD", "KNN", "SVC"]
        df = pd.DataFrame({'Untuned': untuned,'Tuned': tuned}, index=model_types)
        ax = df.plot.bar(rot=0, color={"Untuned": "green", "Tuned": "red"})
        for container in ax.containers:
            ax.bar_label(container)
        ax.set_ylabel('Inference Time [ms]')
        ax.set_title('Model Inference Times (Untuned vs. Tuned)')
        ax.set_ylim(0, np.ceil(max(combined)*1.2))
        plt.show()

    def display_models(self):
        print("DECISION TREE\n")
        print(f"UNTUNED: {self.DT_model.all_models[0]}")
        print(f"TUNED: {self.DT_model.all_models[-1]}\n")

        print("K-NEIGHBOURS CLASSIFIER\n")
        print(f"UNTUNED: {self.KNN_model.all_models[0]}")
        print(f"TUNED: {self.KNN_model.all_models[-1]}\n")

        print("SUPPORT VECTOR MACHINE\n")
        print(f"UNTUNED: {self.SVC_model.all_models[0]}")
        print(f"TUNED: {self.SVC_model.all_models[-1]}\n")

        print("STOCHASTIC GRADIENT DESCENT")
        print(f"UNTUNED: {self.SGD_model.all_models[0]}")
        print(f"TUNED: {self.SGD_model.all_models[-1]}\n")

compare = Compare(dummy_model=dummy, DT_model=dt, KNN_model=knn, SVC_model=svc, SGD_model=sgd)
compare.tabulate()
compare.bar_chart_validation_accuracy()
compare.bar_chart_training_time()
compare.bar_chart_inference_time()
compare.display_models()

print(knn.inference_time)