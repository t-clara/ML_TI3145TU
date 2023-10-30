################################################################################
##                                                                            ##
##                             File: ninefour.py                              ##
##                                                                            ##
##                  Description: Pre-processing, splitting,                   ##
##                 and general information on US census data.                 ##
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
import matplotlib.pyplot as plt
import time

'''
# These are your training samples along with their labels
census_train = np.genfromtxt("data/census_train.csv", delimiter=',', skip_header=1)
census_labels = np.genfromtxt("data/census_labels.csv", delimiter=',', skip_header=1)

# These are unknown instances that you should classify
census_unknown = np.genfromtxt("data/census_unknown.csv", delimiter=',', skip_header=1)

'''
X = pd.read_csv('data\census_train.csv', sep=',', header=0)
y = pd.read_csv('data\census_labels.csv', sep=',', header=0)
#X = pd.read_csv(r"C:\Users\thiba\Downloads\Running\data\census_train.csv", sep=',', header=0)
#y = pd.read_csv(r"C:\Users\thiba\Downloads\Running\data\census_labels.csv", sep=',', header=0)

########################################
#                                      #
#                CLASS                 #
#                                      #
########################################

class USdata:
    '''Class to operate on the 1994 data'''
    def __init__(self, X=X, y=y, random_state: int = 42) -> None:
        '''Initializer to read the data'''
        self.random_state = random_state
        self.X = X
        self.y = y
        self.X_np = np.array(self.X)
        self.y_np = np.array(self.y)
        
    def display(self) -> None:
        '''Displays the whole dataset'''
        print(self.X)
        print('\n and here are the corresponding labels')
        print(self.y)
    
    def unique(self):
        for cat in self.classes:
            print(set(self.X[cat]))
    
    def data_information(self, show_info: bool = True) -> None:
        '''Gathers information for the first classes'''
        self.classes = list(self.X.columns)
        self.classes_dict = {cls: len(self.X[cls]) for cls in self.classes}
        if show_info:
            print(f'There are {len(self.classes)} and each class has the following object distirbution: {self.classes_dict}')
            print('\n Here is further statistical data from the data set')
            print(self.X.describe())

    def remove_nan(self) -> None:
        '''For any objects that include ?, they are deleted'''

        #Removing ? from the data X
        rows_to_drop = self.X[self.X.isin(['?']).any(axis=1)].index
        self.X = self.X.drop(index=rows_to_drop)
        self.y = self.y.drop(index=rows_to_drop)

        #Removing NaN X
        rows_to_drop_nan = self.X[self.X.isna().any(axis=1)].index
        self.X = self.X.drop(index=rows_to_drop_nan)
        self.y = self.y.drop(index=rows_to_drop_nan)

        # Concatinate labels and drop
        train_concat = pd.DataFrame(pd.concat([self.X_train, self.y_train], axis=1))
        test_concat = pd.DataFrame(pd.concat([self.X_train, self.y_test], axis=1))
        cv_concat = pd.DataFrame(pd.concat([self.X_train, self.y_cv], axis=1))

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
        self.X_train, self.y_train = train_concat.iloc[:, :-1], train_concat.iloc[:, -1:]
        self.X_test, self.y_test = test_concat.iloc[:, :-1], test_concat.iloc[:, -1:]
        self.X_cv, self.y_cv = cv_concat.iloc[:, :-1], cv_concat.iloc[:, -1:]

    def check_nan(self, X, y) -> None:
        '''Checking for NaN values'''
        print("STATUS: Checking for NaN values in X and y")

        x_nans, x_nans_number = X.isnull().values.any(), X.isna().sum().sum()
        y_nans, y_nans_number = y.isnull().values.any(), y.isna().sum().sum()

        m1 = f'There are {x_nans_number} NaNs in X' if x_nans else f'There are no NaN in X'
        m2 = f'There are {y_nans_number} NaNs in y' if y_nans else f'There are no NaN in y'
        print(f"INFO: {m1}")
        print(f"INFO: {m2}")

    def preprocess(self, with_mean: bool = True) -> None:
        '''Processes the categorical classes to binary
        and income and sex to binary classes specifically'''
    
        le = LabelEncoder()

        # Changing sex to binary (female vs. male become binary states)
        self.X['sex'] = le.fit_transform(self.X['sex'])

        #Standardize- age, education-num and hours-per-week columns
        #Does not affect the dummy classifier
        sc = StandardScaler(with_mean=with_mean)
        train_col_scale = self.X_train[['age','education-num','hours-per-week']]
        train_col_scale_test = self.X_test[['age','education-num','hours-per-week']]
        train_col_scale_cv = self.X_cv[['age','education-num','hours-per-week']]
        train_scaler_col = sc.fit_transform(train_col_scale)
        train_scaler_col_test = sc.transform(train_col_scale_test)
        train_scaler_col_cv = sc.transform(train_col_scale_cv)

        train_scaler_col = pd.DataFrame(train_scaler_col,columns=train_col_scale.columns)
        train_scaler_col_test = pd.DataFrame(train_scaler_col_test,columns=train_col_scale.columns)
        train_scaler_col_cv = pd.DataFrame(train_scaler_col_cv,columns=train_col_scale.columns)

        # Switch Train
        self.X_train['age']= train_scaler_col['age']
        self.X_train['education-num']= train_scaler_col['education-num']
        self.X_train['hours-per-week']= train_scaler_col['hours-per-week']

        # Switch Test
        self.X_test['age']= train_scaler_col_test['age']
        self.X_test['education-num']= train_scaler_col_test['education-num']
        self.X_test['hours-per-week']= train_scaler_col_test['hours-per-week']

        #Switch CV
        self.X_cv['age']= train_scaler_col_cv['age']
        self.X_cv['education-num']= train_scaler_col_cv['education-num']
        self.X_cv['hours-per-week']= train_scaler_col_cv['hours-per-week']

        #Tabulating the rest of the tables as binary
        # Categorical classes are converted to columns with name
        # of categorical class with True or False value.
        self.X = pd.get_dummies(self.X, drop_first=True)
        self.X_train = pd.get_dummies(self.X, drop_first=True)
        self.X_test = pd.get_dummies(self.X, drop_first=True)
        self.X_cv = pd.get_dummies(self.X, drop_first=True)


    def dummy(self, n_split: int = 10, test_size: float = 0.5, show_average: int = False) -> None:
        '''Construction of a Dummy classifier using CV'''
        cv = ShuffleSplit(n_splits=n_split, test_size=test_size, random_state=self.random_state)
        dummy_classifier = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
        self.cv_results_dummy = cross_validate(dummy_classifier, self.X, self.y, cv=cv, n_jobs=2)
        copy = list(self.cv_results_dummy['test_score'])
        if show_average:
            for i in range(len(copy)):
                print(copy[i])
        else:
            print(np.mean(copy))

    def split(self, test_size: float, cv_size: float | None = None):
        '''Create a training/test split. Alternatively, you can have a validation set'''
        #Reminder for the users
        print("CAUTION: Call .preprocess() before .split() to ensure all features have been standardized.")

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
            
    def get_columns(self):
        if not isinstance(self.X, np.ndarray):
            return list(self.X.columns)
        return f'The type is not Pandas - Get scrubed'
 
    def tonumpy(self) -> None:
        '''Transform everything to Numpy Array'''
        print("CAUTION: Call .tonumpy() after .preprocess() and .split().")

        #For the general data
        self.X = self.X.to_numpy()
        self.y = self.y.to_numpy().ravel()

        #Check if the split is made
        if hasattr(self, 'X_train'):
            if not isinstance(self.X_train, np.ndarray):
                self.X_train = self.X_train.to_numpy()
                self.y_train = self.y_train.to_numpy().ravel()
            if not isinstance(self.X_test, np.ndarray):
                self.X_test = self.X_test.to_numpy()
                self.y_test = self.y_test.to_numpy().ravel()

        if hasattr(self, 'X_cv'):
            if not isinstance(self.X_cv, np.ndarray):
                self.X_cv = self.X_cv.to_numpy()
                self.y_cv = self.y_cv.to_numpy().ravel()
    


'''
def main():
    us = USdata()
    us.data_information(False)
    us.remove_nan()
    #us.data_information()
    #us.display()
    us.preprocess()
    #us.display()
    us.dummy()
    us.split(0.5, 0.2)
    us.check_nan(us.X_subset, us.y_subset)
    us.tonumpy()
    

if __name__ == main():
    main()
'''