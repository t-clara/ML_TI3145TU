################################################################################
##                                                                            ##
##                              File: models.py                               ##
##                                                                            ##
##  Description: Implementation of DecisionTree, KNeighbors, SVC, SGD, PCA    ##
##             alongside tuning sub-functions and plotting tools.             ##
##                                                                            ##
##                Authors: Ali Alper Atasoglu, Thibault Clara                 ##
##                                                                            ##
################################################################################

########################################
#                                      #
#               PACKAGES               #
#                                      #
########################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn import tree
# Import using "pip install alive-progress":
from alive_progress import alive_bar

import time
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import itertools
from beautifultable import BeautifulTable
import pandas as pd

########################################
#                                      #
#                MODELS                #
#                                      #
########################################

class DecisionTreeClassification:
    def __init__(self, X_train, y_train, X_cv, y_cv, max_depth: int = None, min_samples_leaf: int = 2, random_state: int = 42) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_cv = X_cv
        self.y_cv = y_cv
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)
        # Common Attributes used for Comparison:
        self.accuracy_train = None
        self.accuracy_val = None
        self.training_time = None
        self.inference_time = None

    def __str__(self):
        return f"DecisionTreeClassifier(max_depth={self.max_depth}, min_samples_leaf={self.min_samples_leaf}, random_state={self.random_state})"
    
    def optimize(self, max_max_depth: int, max_min_samples_leaf: int) -> None:        
        train_accuracy = []
        cv_accuracy = []
        counter1 = len(range(1, max_max_depth))
        counter2 = len(range(1, max_min_samples_leaf))
        total = counter1 * counter2
        with alive_bar(total) as bar:
            for max_depth in range(1, max_max_depth):
                for min_samples_leaf in range(1, max_min_samples_leaf):
                    local_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=self.random_state)
                    local_model.fit(self.X_train, self.y_train)
                    predictions_train = local_model.predict(self.X_train)
                    predictions_cv = local_model.predict(self.X_cv)
                    train_accuracy.append((accuracy_score(predictions_train, self.y_train), (max_depth, min_samples_leaf)))
                    cv_accuracy.append((accuracy_score(predictions_cv, self.y_cv), (max_depth, min_samples_leaf)))
                    time.sleep(.005)
                    bar()
                
        print(f"STATUS: OPTIMAL MODEL IDENTIFIED!")

        max_depth_optimal = cv_accuracy.index(max(cv_accuracy)) + 1
        self.max_depth, self.min_samples_leaf = cv_accuracy[max_depth_optimal - 1][1]
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)
        
        print(str(self))
        print(f"INFO: Validation Accuracy={cv_accuracy[max_depth_optimal - 1][0]}")
        print(f"INFO: Training Accuracy={train_accuracy[max_depth_optimal - 1][0]}")

        self.accuracy_train = train_accuracy[max_depth_optimal - 1][0]
        self.accuracy_val = cv_accuracy[max_depth_optimal - 1][0]
       
        
    def train(self, X_train, y_train, labels):
        train_time = []
        # Train Time Start
        train_time_start = time.perf_counter()
        # Model Construction
        self.model.fit(X_train, y_train)
        # Train Time Stop
        train_time_stop = time.perf_counter()
        train_time.append(train_time_stop - train_time_start)
        '''
        plt.figure(figsize=(20,20),dpi=80)
        tree.plot_tree(self.model, filled=True, feature_names=labels)
        plt.show()
        '''
        self.training_time = train_time[0]
        # No Predictions
        self.inference_time = None


class KNeighborsClassification:
    def __init__(self, X, y, X_train, y_train, X_test, y_test, X_cv, y_cv, n_neighbors: int = 5, weights: str = 'uniform') -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_cv = X_cv
        self.y_cv = y_cv
        # Common Attributes used for Comparison:
        self.accuracy_train = None
        self.accuracy_val = None
        self.training_time = None
        self.inference_time = None
    
    def __str__(self) -> str:
        return f"KNeighborsClassifier(n_neighbors={self.n_neighbors}, weights={self.weights})"

    def optimize(self, max_n_neighbors: int, n_folds: int = 5) -> None:

        #validation_accuracy = []
        infer_time = []
        n_folds = n_folds

        #Parameter space
        n_neighbors_list = range(1, max_n_neighbors+1)
        weights_list = ['uniform', 'distance', None]


        parameters = {
                'n_neighbors': n_neighbors_list,
                'weights': weights_list,
            }
        # Grid Search Construction
        local_model = KNeighborsClassifier()
        gs = GridSearchCV(local_model, parameters, cv=n_folds, verbose=10, n_jobs=-1)
        infer_time = []
        #Setting up the optimization with GridSearch
        infer_time_start = time.perf_counter()
        gs.fit(self.X_train, self.y_train)
        infer_time_stop = time.perf_counter()
        infer_time.append(infer_time_stop - infer_time_start)
        
        # Optimization Info
        self.optimize_info = {'mean_fit_time': gs.cv_results_['mean_fit_time'], 'mean_score_time': gs.cv_results_['mean_score_time']}

        # Identifying Optimal Model
        best_model = gs.best_estimator_
        best_score = gs.best_score_
        best_param = gs.best_params_
        self.n_neighbors, self.weights = best_param['n_neighbors'], best_param['weights']

        #Printing the result
        print(f"STATUS: OPTIMAL MODEL IDENTIFIED!")
        print(f'INFO: Setting the best model as a class attribute')
        self.model = best_model
        print(f"INFO: {str(self)}")
        print(f"INFO: Optimal Model Score = {best_score} with {n_folds}-fold cross-validation.")
        print(f"INFO: The model took {np.mean(infer_time)} seconds to optimize, the mean fit time was\
            {np.mean(self.optimize_info['mean_fit_time'])} and the\
            mean score time was {np.mean(self.optimize_info['mean_score_time'])}")
        print('===========================\n')
        
        '''
        for n_neighbors in range(1, max_n_neighbors+1):
            # Local Model Construction
            local_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=self.weights)
            # Accuracy Assessment 
            cross_validation_accuracy = cross_val_score(local_model, self.X, self.y, cv=n_folds)
            cv_accuracy.append(np.mean(cross_validation_accuracy))
            K_optimal = cv_accuracy.index(max(cv_accuracy)) + 1
            self.n_neighbors = K_optimal
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        
        #Plotting
        plt.title(f"Validation Accuracy for Various K for weights={self.weights}")
        plt.plot(np.arange(1, max_n_neighbors+1), validation_accuracy, label=f"Average Cross-Validation Set with n_folds={n_folds}")
        plt.plot(self.n_neighbors, validation_accuracy[self.n_neighbors - 1], marker="X")
        plt.xlabel("K")
        plt.ylabel(f"Average Cross-Validation Set Accuracy with n_folds={n_folds}")
        plt.legend()
        plt.show()

        # Plotting Inference Time:

        plt.title('Inference Time')
        plt.plot(np.arange(1, max_n_neighbors+1), infer_time)
        plt.xlabel('K')
        plt.ylabel('Inference Time [s]')
        plt.show()
        '''

    def train(self):
        print(f"CAUTION: You have just called .train() for KNeighborsClassification, make sure that you fed the training data.")
        print(f"STATUS: Training for {str(self)}...")

        print('\n===========================')
        input_val = input('Enter True to check with CV, False to check with Test\n')
        print('===========================\n')

        if input_val:
            infer_time = []
            # Infer Time Start
            infer_time_start = time.perf_counter()
            # Model Construction
            self.model.fit(self.X_train, self.y_train)
            # Infer Time Stop
            infer_time_stop = time.perf_counter()
            infer_time.append(infer_time_stop - infer_time_start)
            # Testing Data:
            cv_predictions = self.model.predict(self.X_cv)
            cv_accuracy = accuracy_score(self.y_cv, cv_predictions)
            print(f'INFO: Accuracy (KNeighborsClassification) = {cv_accuracy}')
            infer_time_average = np.mean(infer_time)
            print(f'INFO: Average Inference Time (KNeighborsClassification) = {infer_time_average}')
        else:
            infer_time = []
            # Infer Time Start
            infer_time_start = time.perf_counter()
            # Model Construction
            self.model.fit(self.X_train, self.y_train)
            # Infer Time Stop
            infer_time_stop = time.perf_counter()
            infer_time.append(infer_time_stop - infer_time_start)
            # Testing Data:
            test_predictions = self.model.predict(self.X_test)
            test_accuracy = accuracy_score(self.y_test, test_predictions)
            print(f'INFO: Accuracy (KNeighborsClassification) = {test_accuracy}')
            infer_time_average = np.mean(infer_time)
            print(f'INFO: Average Inference Time (KNeighborsClassification) = {infer_time_average}')

        self.training_time = train_time[0]
        self.inference_time = infer_time_average


class SVCClassification:
    def __init__(self, X, y, X_train, y_train, X_test, y_test, X_cv, y_cv, C: int = 10, kernel: str = 'poly', degree: int = 3, gamma: str = 'scale', random_state: int = 42) -> None:
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_cv = X_cv
        self.y_cv = y_cv
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state
        self.model = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, random_state=self.random_state)
        # Common Attributes used for Comparison:
        self.accuracy_train = None
        self.accuracy_val = None
        self.training_time = None
        self.inference_time = None
    
    def __str__(self) -> str:
        return f"SVC(C={self.C}, kernel={self.kernel}, degree={self.degree}, gamma={self.gamma}, random_state={self.random_state})"
    
    def optimize(self, n_folds: int = 5, further_optimize: bool = False) -> None:
        if further_optimize:
            #Parametrization space
            optimal_parameters = self.model.get_params()
            #Tuning parameters
            margin = 0.20
            n_points = 5
            C_lower, C_upper = (1-margin) * optimal_parameters['C'], (1+margin) * optimal_parameters['C']
            C_list = np.linspace(C_lower, C_upper, n_points).tolist()
            kernel_list = [optimal_parameters['kernel']]
            degree_lower, degree_upper = (1-margin) * optimal_parameters['degree'], (1+margin) * optimal_parameters['degree']
            degree_list = [int(x) for x in np.linspace(degree_lower, degree_upper, n_points)]
            gamma_list = [optimal_parameters['gamma']]

            parameters = {
                'C': C_list,
                'kernel': kernel_list,
                'degree': degree_list,
                'gamma': gamma_list,
                'random_state': [self.random_state]
            }

            # Grid Search Construction
            local_model = SVC()
            gs = GridSearchCV(local_model, parameters, cv=n_folds, verbose=10, n_jobs=-1)
            infer_time = []
            #Setting up the optimization with GridSearch
            infer_time_start = time.perf_counter()
            gs.fit(self.X_train, self.y_train)
            infer_time_stop = time.perf_counter()
            infer_time.append(infer_time_stop - infer_time_start)
            
            # Optimization Info
            self.optimize_info = {'mean_fit_time': gs.cv_results_['mean_fit_time'], 'mean_score_time': gs.cv_results_['mean_score_time']}

            # Identifying Optimal Model
            best_model = gs.best_estimator_
            best_score = gs.best_score_
            best_param = gs.best_params_
            self.C, self.kernel, self.degree, self.gamma = best_param['C'], best_param['kernel'], best_param['degree'], best_param['gamma']

            #Printing the result
            print(f"STATUS: OPTIMAL MODEL IDENTIFIED!")
            print(f'INFO: Setting the best model as a class attribute')
            self.model = best_model
            print(f"INFO: {str(self)}")
            print(f"INFO: Optimal Model Score = {best_score} with {n_folds}-fold cross-validation.")
            print(f"INFO: The model took {np.mean(infer_time)} seconds to optimize, the mean fit time was\
                {np.mean(self.optimize_info['mean_fit_time'])} and the\
                mean score time was {np.mean(self.optimize_info['mean_score_time'])}")
            print('===========================\n')

        else:
            # Parametrization Space
            #C_list = np.logspace(-1, 2, num=4).tolist()
            #kernel_list = ['linear', 'poly']
            #degree_list = np.arange(1, 10)
            #gamma_list = ['scale', 'auto']

            C_list = [0.1]
            kernel_list = ['linear', 'poly']
            degree_list = np.arange(1, 2)
            gamma_list = ['scale']


            # Dictionary of Parametrization Space
            parameters = {
                'C': C_list,
                'kernel': kernel_list,
                'degree': degree_list,
                'gamma': gamma_list,
                'random_state': [self.random_state]
            }

            # Grid Search Construction
            local_model = SVC()
            gs = GridSearchCV(local_model, parameters, cv=n_folds, verbose=10, n_jobs=-1)
            infer_time = []
            #Setting up the optimization with GridSearch
            infer_time_start = time.perf_counter()
            gs.fit(self.X_train, self.y_train)
            infer_time_stop = time.perf_counter()
            infer_time.append(infer_time_stop - infer_time_start)
            
            # Optimization Info
            self.optimize_info = {'mean_fit_time': gs.cv_results_['mean_fit_time'], 'mean_score_time': gs.cv_results_['mean_score_time']}

            # Identifying Optimal Model
            best_model = gs.best_estimator_
            best_score = gs.best_score_
            best_param = gs.best_params_
            self.C, self.kernel, self.degree, self.gamma = best_param['C'], best_param['kernel'], best_param['degree'], best_param['gamma']

            #Printing the result
            print(f"STATUS: OPTIMAL MODEL IDENTIFIED!")
            print(f'INFO: Setting the best model as a class attribute')
            self.model = best_model
            print(f"INFO: {str(self)}")
            print(f"INFO: Optimal Model Score = {best_score} with {n_folds}-fold cross-validation.")
            print(f"INFO: The model took {np.mean(infer_time)} seconds to optimize, the mean fit time was\
                {np.mean(self.optimize_info['mean_fit_time'])} and the\
                mean score time was {np.mean(self.optimize_info['mean_score_time'])}")
            print('===========================\n')
        
    
    def train(self) -> None:
        print(f"CAUTION: You have just called .train() for SVCClassification, make sure that you fed the training data.")
        print(f"STATUS: Training for {str(self)}...")

        print('\n===========================')
        input_val = input('Enter True to check with CV, False to check with Test\n')
        print('===========================\n')

        print('\n===========================')
        input_val = input('Enter True to check with CV, False to check with Test\n')
        print('===========================\n')

        train_time = []
        # Train Time Start
        train_time_start = time.perf_counter()
        # Model Construction
        self.model.fit(self.X_train, self.y_train)
        # Training Time Stop
        training_time_stop = time.perf_counter()
        training_time.append(training_time_stop - training_time_start)

        if input_val:
            infer_time = []
            # Infer Time Start
            infer_time_start = time.perf_counter()
            # Prediction
            cv_predictions = self.model.predict(self.X_cv)
            # Infer Time Stop
            infer_time_stop = time.perf_counter()
            infer_time.append(infer_time_stop - infer_time_start)

            print(f"STATUS: *** CHECKING ACCURACY ON VALIDATION SET ***")
            cv_accuracy = accuracy_score(self.y_cv, cv_predictions)
            
            # Report Performance
            print(f'INFO: Accuracy (SVCClassification) = {cv_accuracy}')
            training_time_average = np.mean(training_time)
            infer_time_average = np.mean(infer_time)
            print(f"INFO: Average Training Time (SVCClassification) = {training_time_average}")
            print(f'INFO: Average Inference Time (SVCClassification) = {infer_time_average}')

            self.training_time = training_time_average
            self.inference_time = infer_time_average
            self.accuracy_test = cv_accuracy
            self.error_rate_test = 1-cv_accuracy

            SVC_table = BeautifulTable()
            SVC_table.columns.header = ["", "Average Training Time","Average Inference Time","Validation Accuracy"]
            SVC_table.rows.append(['Optimal SVC', training_time_average, infer_time_average, cv_accuracy])
            print(SVC_table)
        else:
            infer_time = []
            # Infer Time Start
            infer_time_start = time.perf_counter()
            # Prediction
            test_predictions = self.model.predict(self.X_test)
            # Infer Time Stop
            infer_time_stop = time.perf_counter()
            infer_time.append(infer_time_stop - infer_time_start)

        train_accuracy = accuracy_score(self.y_train, train_predictions)
        cv_accuracy = accuracy_score(self.y_cv, cv_predictions)

        self.accuracy_train = train_accuracy
        self.accuracy_val = cv_accuracy
         
        # Report Performance
        print(f'INFO: Training Accuracy (SVCClassification) = {train_accuracy}')
        print(f'INFO: Validation Accuracy (SVCClassification) = {cv_accuracy}')
        
        training_time_average = np.mean(train_time)
        infer_time_average = np.mean(infer_time)

        self.training_time = training_time_average
        self.inference_time = infer_time_average
        print(f"INFO: Average Training Time (SVCClassification) = {training_time_average}")
        print(f'INFO: Average Inference Time (SVCClassification) = {infer_time_average}')

        self.training_time = training_time_average
        self.inference_time = infer_time_average

        SVC_table = BeautifulTable()
        SVC_table.columns.header = ["", "Average Training Time","Average Inference Time","Training Accuracy", "Validation Accuracy"]
        SVC_table.rows.append(['Optimal SVC', training_time_average, infer_time_average, train_accuracy, cv_accuracy])
        print(SVC_table)
        
class SGDClassification:
    '''SGD Classifier Common Class'''
    def __init__(self, X, y, X_train, y_train, X_test, y_test, X_cv, y_cv, loss: str = 'log_loss', 
                 penalty: str | None = 'l2', alpha: float = 0.0001, 
                 max_iter: int = 1000, learning_rate: str = 'constant', eta0: float = 0.1, 
                 random_state: int = 42, warm_start: bool = True) -> None:
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_cv = X_cv
        self.y_cv = y_cv
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.random_state = random_state
        self.warm_start = warm_start
        self.model = SGDClassifier(loss=self.loss, penalty=self.penalty, alpha=self.alpha, max_iter=self.max_iter, \
                                    random_state=self.random_state, learning_rate=self.learning_rate, \
                                      eta0=self.eta0, warm_start=self.warm_start)
        self.cv_accuracy: list[float] = []
        self.optimal_models = []
        # Common Attributes used for Comparison:
        self.accuracy_train = None
        self.accuracy_val = None
        self.training_time = None
        self.inference_time = None
    
    def __str__(self) -> str:
        '''Str representation of SGD'''
        return f'SGD with alpha = {self.alpha}, eta0 = {self.eta0}, penalty = {self.penalty}, max_iter = {self.max_iter}'
    
    def optimize(self, further_optimize: bool = False):
        '''Optimizes the SGD model'''
        if further_optimize:
            optimal_parameters = self.model.get_params()
            #Tuning parameters
            margin = 0.20
            n_points = 5
            alpha_lower, alpha_upper = (1-margin) * optimal_parameters['alpha'], (1+margin) * optimal_parameters['alpha']
            alpha_list = np.linspace(alpha_lower, alpha_upper, n_points).tolist()
            eta0_lower, eta0_upper = (1-margin) * optimal_parameters['eta0'], (1+margin) * optimal_parameters['eta0']
            eta0_list = np.linspace(eta0_lower, eta0_upper, n_points).tolist()
            penalty_list = [optimal_parameters['penalty']]
            max_iter_lower, max_iter_upper = (1-margin) * optimal_parameters['max_iter'], (1+margin) * optimal_parameters['max_iter']
            max_iter_list = [int(x) for x in np.linspace(max_iter_lower, max_iter_upper, n_points)]

            sgdgs_params = {
            'alpha': alpha_list,
            'eta0': eta0_list,
            'penalty': penalty_list,
            'max_iter': max_iter_list,
            'loss': ['log_loss'],
            'random_state': [self.random_state],
            'learning_rate': [self.learning_rate],
            'warm_start': [self.warm_start]
            }
            
            #Local sgd and grid search algorithms
            local_sgd = SGDClassifier()
            gs = GridSearchCV(local_sgd, sgdgs_params, cv=5, verbose=10, n_jobs=-1,  error_score='raise', return_train_score=True)

            infer_time = []
            #Setting up the optimization with GridSearch
            infer_time_start = time.perf_counter()
            gs.fit(self.X_train, self.y_train)
            infer_time_stop = time.perf_counter()
            infer_time.append(infer_time_stop - infer_time_start)

            #Optimization info
            self.optimize_info = {'mean_fit_time': gs.cv_results_['mean_fit_time'], 'mean_score_time': gs.cv_results_['mean_score_time']}

            #Finding the most optimal
            best_model = gs.best_estimator_
            best_score = gs.best_score_
            best_param = gs.best_params_

            #Printing the result
            print('\n===========================')
            self.penalty, self.alpha, self.max_iter, self.eta0 = best_param['penalty'], best_param['alpha'], best_param['max_iter'], best_param['eta0']
            self.model = best_model
            self.optimal_models.append(best_param)
            print(f"INFO: {str(self)}")
            print(f'INFO: The model scored {best_score} with a 5 split CV')
            print(f"INFO: The model took {np.mean(infer_time)} seconds to optimize, the mean fit time was\
                {np.mean(self.optimize_info['mean_fit_time'])} and the\
                mean score time was {np.mean(self.optimize_info['mean_score_time'])}")
            print(f'INFO: Setting the best model as a class attribute')
            print('===========================\n')
        else:
            #Tuning parameters
            alpha_list = np.logspace(-4, 4, num=9).tolist() + [0]
            eta0_list = np.logspace(-4, 4, num=9).tolist()
            penalty_list = ['l1', 'l2', None]
            max_iter_list = [int(x) for x in np.logspace(2, 5, num=4).tolist()]

            sgdgs_params = {
            'alpha': alpha_list,
            'eta0': eta0_list,
            'penalty': penalty_list,
            'max_iter': max_iter_list,
            'loss': ['log_loss'],
            'random_state': [self.random_state],
            'learning_rate': [self.learning_rate],
            'warm_start': [self.warm_start]
            }
            
            #Local sgd and grid search algorithms
            local_sgd = SGDClassifier()
            gs = GridSearchCV(local_sgd, sgdgs_params, cv=5, verbose=10, n_jobs=-1,  error_score='raise', return_train_score=True)

            infer_time = []
            #Setting up the optimization with GridSearch
            infer_time_start = time.perf_counter()
            gs.fit(self.X_train, self.y_train)
            infer_time_stop = time.perf_counter()
            infer_time.append(infer_time_stop - infer_time_start)

            #Optimization info
            self.optimize_info = {'mean_fit_time': gs.cv_results_['mean_fit_time'], 'mean_score_time': gs.cv_results_['mean_score_time']}

            #Finding the most optimal
            best_model = gs.best_estimator_
            best_score = gs.best_score_
            best_param = gs.best_params_

            #Printing the result
            print('\n===========================')
            print(f"STATUS: OPTIMAL MODEL IDENTIFIED!")
            self.penalty, self.alpha, self.max_iter, self.eta0 = best_param['penalty'], best_param['alpha'], best_param['max_iter'], best_param['eta0']
            self.model = best_model
            self.optimal_models.append(best_param)
            print(f"INFO: {str(self)}")
            print(f'INFO: The model scored {best_score} with a 5 split CV')
            print(f"INFO: The model took {np.mean(infer_time)} seconds to optimize, the mean fit time was\
                {np.mean(self.optimize_info['mean_fit_time'])} and the\
                mean score time was {np.mean(self.optimize_info['mean_score_time'])}")
            print(f'INFO: Setting the best model as a class attribute')
            
            print('===========================\n')

    def train(self, n_batches: int = 300, show_loss: bool = True, show_acc: bool = False) -> None:
        '''Train the SGD classifier with the best parameter'''
        
        print('\n===========================')
        input_val = input('Enter True to check with CV, False to check with Test\n')
        print('===========================\n')

        #Disclaimers
        print(f"CAUTION: You have just called .train() for SGDClassification, make sure that you fed the training data.")
        print(f"STATUS: Training for {str(self)}...")

        if input_val == 'True':
            infer_time, sgd_train_loss, sgd_other_loss = [], [], []
            sgd_train_score, sgd_other_score = [], []
            # Infer Time Start
            train_time = []
            infer_time = []
            with alive_bar(len(range(n_batches))) as bar:
                for _ in range(n_batches):
                    # Train Time Start
                    train_time_start = time.perf_counter()

                    # Model Construction
                    self.model.partial_fit(self.X_train, self.y_train, classes = np.unique(self.y_train))
                    
                    # Train Time Stop
                    train_time_stop = time.perf_counter()
                    train_time.append(train_time_stop - train_time_start)
                    
                    # Infer Time Start
                    infer_time_start = time.perf_counter()
                    # Predictions:
                    train_predictions = self.model.predict(self.X_train)
                    train_predictions_proba = self.model.predict_proba(self.X_train)
                    cv_predictions = self.model.predict(self.X_cv)
                    cv_predictions_proba = self.model.predict_proba(self.X_cv)
                    # Infer Time Stop
                    infer_time_stop = time.perf_counter()
                    infer_time.append(infer_time_stop - infer_time_start)

                    ### DID NOT IMPLEMENT LOSS & SCORE FOR CV ###

                    sgd_train_loss.append(log_loss(self.y_train, train_predictions_proba))
                    sgd_train_score.append(1 - accuracy_score(self.y_train, train_predictions))
                    
                    train_accuracy = accuracy_score(self.y_train, train_predictions)
                    cv_accuracy = accuracy_score(self.y_cv, cv_predictions)

                    # Other Data
                    other_predictions = self.model.predict(self.X_cv)
                    other_predictions_proba = self.model.predict_proba(self.X_cv)

                    sgd_other_loss.append(log_loss(self.y_cv, other_predictions_proba))
                    sgd_other_score.append(1 - accuracy_score(self.y_cv, other_predictions))

                    #Bar settings
                    time.sleep(.005)
                    bar()

            infer_time_average = np.mean(infer_time)
            print(f'INFO: Mean Accuracy (SGDClassification) = {1 - np.mean(sgd_other_score)}')
            print(f'INFO: Average Inference Time (SGDClassification) = {infer_time_average}')

            if show_loss:
                plt.plot(sgd_train_loss, label = 'Train Log Loss', color='blue')
                plt.plot(sgd_other_loss, label = 'Validation Log Loss', color='red')
                plt.xlabel('Epoch Number')
                plt.ylabel('Log Losses')
                plt.title('Log Loss vs Number of Epochs')
                plt.legend()
                plt.show()
            if show_acc:
                plt.plot(sgd_train_score, label = 'Train Error Rate', color='blue')
                plt.plot(sgd_other_score, label = 'Validation Error Rate', color='green')
                plt.xlabel('Epoch Number')
                plt.ylabel('Error Rate')
                plt.title('Error Rate vs Number of Epochs')
                plt.legend()
                plt.show()
        else:
            infer_time, sgd_train_loss, sgd_other_loss = [], [], []
            sgd_train_score, sgd_other_score = [], []
            # Infer Time Start
            with alive_bar(len(range(n_batches))) as bar:
                for _ in range(n_batches):
                    infer_time_start = time.perf_counter()

                    # Model Construction
                    self.model.partial_fit(self.X_train, self.y_train, classes = np.unique(self.y_train))
                    
                    # Infer Time Stop
                    infer_time_stop = time.perf_counter()
                    infer_time.append(infer_time_stop - infer_time_start)
                    
                    # Training Data:
                    train_predictions = self.model.predict(self.X_train)
                    train_predictions_proba = self.model.predict_proba(self.X_train)

                    sgd_train_loss.append(log_loss(self.y_train, train_predictions_proba))
                    sgd_train_score.append(1 - accuracy_score(self.y_train, train_predictions))

                    # Other Data
                    other_predictions = self.model.predict(self.X_test)
                    other_predictions_proba = self.model.predict_proba(self.X_test)

                    sgd_other_loss.append(log_loss(self.y_test, other_predictions_proba))
                    sgd_other_score.append(1 - accuracy_score(self.y_test, other_predictions))

                    #Bar settings
                    time.sleep(.005)
                    bar()

            infer_time_average = np.mean(infer_time)
            print(f'INFO: Mean Accuracy (SGDClassification) = {1 - np.mean(sgd_other_score)}')
            print(f'INFO: Average Inference Time (SGDClassification) = {infer_time_average}')

            if show_loss:
                plt.plot(sgd_train_loss, label = 'Train Log Loss', color='blue')
                plt.plot(sgd_other_loss, label = 'Test Log Loss', color='red')
                plt.xlabel('Epoch Number')
                plt.ylabel('Log Losses')
                plt.title('Log Loss vs Number of Epochs')
                plt.legend()
                plt.show()
            if show_acc:
                plt.plot(sgd_train_score, label = 'Train Error Rate', color='blue')
                plt.plot(sgd_other_score, label = 'Test Error Rate', color='red')
                plt.xlabel('Epoch Number')
                plt.ylabel('Error Rate')
                plt.title('Error Rate vs Number of Epochs')
                plt.legend()
                plt.show()

    def display(self):
        SGD_table = BeautifulTable()
        SGD_table.columns.header = ["Model #", "CV Accuracy [%]", "penalty", "alpha", "max_iter", "eta0"]
        SGD_table.rows.append(['Optimal SGD 1', self.cv_accuracy[0], f"{self.optimal_models[0]['penalty']}", f"{'{:.2e}'.format(self.optimal_models[0]['alpha'])}", self.optimal_models[0]['max_iter'], self.optimal_models[0]['eta0']])
        SGD_table.rows.append(['Optimal SGD 2', self.cv_accuracy[1], f"{self.optimal_models[1]['penalty']}", f"{'{:.2e}'.format(self.optimal_models[1]['alpha'])}", self.optimal_models[1]['max_iter'], self.optimal_models[1]['eta0']])
        print(SGD_table)    

        print(f"alpha_1 = {'{:.2e}'.format(self.optimal_models[0]['alpha'])}")
        print(f"alpha_2 = {'{:.2e}'.format(self.optimal_models[1]['alpha'])}")


class Data_PCA:
    def __init__(self, X, y, X_train, y_train,  X_test, y_test, X_cv, y_cv, n_components: int = 10, whiten = True, random_state: int = 42, threshold: float = 0.95) -> None:
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_cv = X_cv
        self.y_cv = y_cv
        self.n_components  = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.threshold = threshold
        self.changed = False

    def plot_components(self) -> None:
        from sklearn.decomposition import PCA
        #Building the model
        self.pca = PCA(self.threshold, whiten=self.whiten, random_state=self.random_state)
        self.pca.fit(self.X)

        #Finding the Elbow
        self.changed = True
        self.copy_explained = self.pca.explained_variance_ratio_
        self.components = self.pca.components_
        variance_summed = [sum(self.copy_explained[:i]) for i in range(1, len(self.copy_explained))]
        plt.title("PCA Analysis")
        plt.xlabel("Number of Principal Components (Descending Order of Importance)")
        plt.ylabel("Cumulative Sum of PC's")
        plt.title('PCA Cumulative Sum with Threshold')
        plt.plot(range(1, len(self.components)), variance_summed, 'or--')
        plt.axhline(y = self.threshold, color = 'b', linestyle = ':', label=f'Threshold={self.threshold}') 
        plt.legend()
        plt.show()
    
    def change_data(self, n_components: int) -> None:
        from sklearn.decomposition import PCA
        if self.changed:
            print(f'CAUTION: You are running n_components = {n_components} and not the class variable n_component = {self.n_components}')

            pca = PCA(n_components, whiten=self.whiten, random_state=self.random_state)
            #Setting and transforming
            self.X_train = pca.fit(self.X_train).transform(self.X_train)
            self.X_test = pca.transform(self.X_test)
            self.X_cv = pca.transform(self.X_cv)

            #Confirmation
            copy_explained = pca.explained_variance_ratio_
            print(f'STATUS: Successfully fit and transformed the data set with explained variance of {sum(copy_explained)}')


            '''
            #Get the index of most important feature
            descending_order = [np.argsort(np.abs(pca.components_[i])).tolist()[::-1] for i in range(n_pcs)]
            print(len(descending_order))
            most_imp_names = [[initial_labels[i] for i in descending_order[j]] for j in range(n_pcs)]
            print([x[:3] for x in most_imp_names])
            '''

        else:
            print(f'CAUTION: You have not run plot components() yet! Visually confirm the elbow first.')

    def __str__(self):
    
    def preprocess(self, with_mean: bool = True) -> None:
        '''Processes the categorical classes to binary
        and income and sex to binary classes specifically'''

        le = LabelEncoder()

        # Changing sex to binary (female vs. male become binary states)
        self.X['sex'] = le.fit_transform(self.X['sex'])

        #Standardize- age, education-num and hours-per-week columns
        #Does not affect the dummy classifier
        self.X = pd.get_dummies(self.X, drop_first=True)
        self.X_train = pd.get_dummies(self.X, drop_first=True)
        self.X_test = pd.get_dummies(self.X, drop_first=True)
        self.X_cv = pd.get_dummies(self.X, drop_first=True)

    def remove_nan(self) -> None:
        '''For any objects that include ?, they are deleted'''

        # Concatinate labels and drop
        total_concat = pd.DataFrame(pd.concat([self.X, self.y], axis=1))
        train_concat = pd.DataFrame(pd.concat([self.X_train, self.y_train], axis=1))
        test_concat = pd.DataFrame(pd.concat([self.X_train, self.y_test], axis=1))
        cv_concat = pd.DataFrame(pd.concat([self.X_train, self.y_cv], axis=1))

        #Removing ? from the data X
        rows_to_drop = total_concat[total_concat.isin(['?']).any(axis=1)].index
        total_concat = total_concat.drop(index=rows_to_drop)

        #Removing NaN X
        rows_to_drop_nan = total_concat[total_concat.isna().any(axis=1)].index
        total_concat = total_concat.drop(index=rows_to_drop_nan)

        #Removing ? from the data x_train
        rows_to_drop_train = train_concat[train_concat.isin(['?']).any(axis=1)].index
        train_concat = train_concat.drop(index=rows_to_drop_train)

        #Removing NaN Train
        rows_to_drop_nan_train = train_concat[train_concat.isna().any(axis=1)].index
        train_concat = train_concat.drop(index=rows_to_drop_nan_train)


        #Removing ? from the data Test
        rows_to_drop_test = test_concat[test_concat.isin(['?']).any(axis=1)].index
        test_concat = test_concat.drop(index=rows_to_drop_test)

        #Removing NaN Test
        rows_to_drop_nan_test = test_concat[test_concat.isna().any(axis=1)].index
        test_concat = test_concat.drop(index=rows_to_drop_nan_test)

        #Removing ? from the data CV
        rows_to_drop_cv = cv_concat[cv_concat.isin(['?']).any(axis=1)].index
        cv_concat = cv_concat.drop(index=rows_to_drop_cv)

        #Removing NaN CV
        rows_to_drop_nan_cv = cv_concat[cv_concat.isna().any(axis=1)].index
        cv_concat = cv_concat.drop(index=rows_to_drop_nan_cv)

        # Re-setting the variables
        self.X, self.y = total_concat.iloc[:, :-1], total_concat.iloc[:, -1:]
        self.X_train, self.y_train = train_concat.iloc[:, :-1], train_concat.iloc[:, -1:]
        self.X_test, self.y_test = test_concat.iloc[:, :-1], test_concat.iloc[:, -1:]
        self.X_cv, self.y_cv = cv_concat.iloc[:, :-1], cv_concat.iloc[:, -1:]