from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import os
import time

class ML_Wrapper(object):
    """This class is a wrapper for the classical machine learning models. We can use this class to train and predict the labels for the test data.
    This class is defined as like deep learning models in order to use the same functions for both models. 

    Args:
        object (_type_): _description_
    """
    def __init__(self, model_name : str, save_path : str, **kwargs):
        """Constructor for the ML_Wrapper class"""
        self.model_name = model_name
        self.save_path = save_path
        self.time_str = time.strftime("%Y%m%d%H%M")
        self._create_checkpoints_directorie()
        self.set_model(kwargs)

    def _create_checkpoints_directorie(self):
        """
        Create Checkpoint directory based on time and architecture mode.
        self.checkdir is going to be initialized. It contains the folder where the checkpoints are going to be saved
        """

        print("Creating checkpoints directories")
        self.checkdir = "{}/checkpoints/{}".format(
            self.save_path, self.time_str + "_" + self.model_name
        )
        if not os.path.exists(self.checkdir):
            os.makedirs(self.checkdir)
        
    def set_model(self, kwargs):
        """Sets the model attribute based on the model_name attribute

        Raises:
            NotImplementedError: Raise the error if the model is not implemented
        """
        if self.model_name == 'svm':
            self.clf = SVC(**kwargs)
        elif self.model_name == 'rf':
            self.clf = RandomForestClassifier(**kwargs)
        elif self.model_name == 'xgb':
            self.clf = XGBClassifier()
        elif self.model_name == 'knn':
            self.clf = KNeighborsClassifier(**kwargs)
        else:
            raise NotImplementedError("Model not implemented")
        
    def train(self, X_train : np.ndarray, y_train : np.ndarray):
        """Trains the classical model

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
        """
        self.clf.fit(X_train, y_train)

        with open(os.path.join(self.checkdir, self.model_name + '.pkl'),'wb') as f:
            pickle.dump(self.clf,f)
        
    def predict(self, X_test : np.ndarray):
        """Predicts the labels for the test data

        Args:
            X_test (np.ndarray): Test data

        Returns:
            np.ndarray: Predicted labels
        """
        return self.clf.predict(X_test)