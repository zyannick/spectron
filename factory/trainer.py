import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from models.classical_ml.ml_wrapper import ML_Wrapper
from models.neural_network.nn_wrapper import NeuralNetworkTrainer
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import random
import torch
import numpy as np
from data_factory.data_reader import BaseDataset
import mlflow

import torch.backends.cudnn as cudnn

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import json

from utils.data_augmentations import Composer, Gaussian_Noise, Poisson_Noise, Wavelength_Shift

from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from pypdf import PdfReader

class Trainer:
    def __init__(self, config_file : str = './configs/fcn_config.yaml', config_custom : dict = {}):
        # Initialize any necessary variables or resources here
        with open(config_file) as f:
            self.config = yaml.load(f, Loader=SafeLoader)

        print(json.dumps(self.config, indent=4, sort_keys=True))

    @property
    def model_type(self):
        return self.config['model']['type']
    
    @property
    def model_name(self):     
        return self.config['model']['name']
    
    @property
    def data_path(self):
        return self.config['dataset']['path']

    @property
    def experiment_path(self):
        return os.path.join(self.config['save_path'], self.config['experiment_name'])
    
    def __init_mlflow(self):
        """Initialize parameters of each run for mlflow and save all the parameters in mlflow.csv.
        For each run, we keep track of :
            - data_path
            - checkpoints_path
            - num_samples_train
            - num_samples_val
            - All the parameters in config_train
        """
        print("Initializing logs for mlflow")
        

        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        def recursive_log_params(config, prefix=''):
            for key, value in config.items():
                if isinstance(value, dict):
                    recursive_log_params(value, prefix + key + '.')
                else:
                    mlflow.log_param(prefix + key, value)

        for key, value in self.config.items():
            try:
                if isinstance(value, dict):
                    recursive_log_params(value, key + '.')
                else:
                    mlflow.log_param(key, value)
            except (
                Exception
            ) as err:  # This error can happen when the value is too long and mlflow refuses to log it
                print("Couldn't log {} because of {}".format(key, err))
        mlflow.log_param("data_path", self.experiment_path)
        mlflow.log_param("num_samples_train", self.train_data.shape[0])
        mlflow.log_param("num_samples_val", self.val_data.shape[0])
        mlflow.log_param("num_samples_test", self.test_data.shape[0])
        runs = mlflow.search_runs()
        runs_csv_file = runs.to_csv( os.path.join(self.experiment_path, 'mlflow.csv')  )

    

    def load_data(self):
        """Loads the data from the specified path and splits it into train, validation and test sets based on the split ratios in the config file"""
        self.data = pd.read_csv(self.data_path, sep=',')

        # remove nan values
        self.data = self.data.dropna()
        # shuffle the data for the splitting
        self.data = self.data.sample(frac=1)

        self.list_of_classes = sorted(self.data['Class'].unique())
        self.classes_to_idx = {self.list_of_classes[i]: i for i in range(len(self.list_of_classes))}
        self.nb_classes = len(self.list_of_classes)
        nb_rows = self.data.shape[0]
        train_split = self.config['dataset']['split']['train']
        val_split = self.config['dataset']['split']['val']

        self.train_data = self.data.iloc[:int(nb_rows*train_split), :]
        self.val_data = self.data.iloc[int(nb_rows*train_split):int(nb_rows*(train_split+val_split)), :]
        self.test_data = self.data.iloc[int(nb_rows*(train_split+val_split)):, :]



    def preprocess_data(self):
        # Preprocess the training data

        # Get the train, validation and test data
        self.x_train = self.train_data.drop(columns=['Class']).to_numpy()
        self.y_train = [ self.classes_to_idx[classe] for classe in self.train_data['Class'].to_list() ]
        self.x_val = self.val_data.drop(columns=['Class']).to_numpy()
        self.y_val = [ self.classes_to_idx[classe] for classe in self.val_data['Class'].to_list() ]
        self.x_test = self.test_data.drop(columns=['Class']).to_numpy()
        self.y_test = [ self.classes_to_idx[classe] for classe in self.test_data['Class'].to_list() ]

        self.x_train_scaled = None
        self.x_val_scaled = None
        self.x_test_scaled = None

        if self.config['dataset'].get('baseline_correction', False):
            # Apply the baseline correction
            if self.config['dataset']['baseline_correction_type'] == 'savgol':
                # Apply the savitzky golay filter
                for i in range(self.x_train.shape[0]):
                    self.x_train[i] = savgol_filter(self.x_train[i], 20, 2)
                for i in range(self.x_val.shape[0]):
                    self.x_val[i] = savgol_filter(self.x_val[i], 20, 2)
                for i in range(self.x_test.shape[0]):
                    self.x_test[i] = savgol_filter(self.x_test[i], 20, 2)

        # Apply the transformations to the data before fitting the selected model
        if self.config['dataset'].get('normalize', False):
            # Normalize the data
            # here we do not normalize
            if self.config['dataset']['normalize_type'] == 'scale':
                # Scale the data
                self.scaler = StandardScaler()
                self.x_train_scaled = self.scaler.fit_transform(self.x_train)
                if len(self.x_val) > 0:
                    self.x_val_scaled = self.scaler.transform(self.x_val)
                self.x_test_scaled = self.scaler.transform(self.x_test)


        composer = None
        if self.config['dataset'].get('augment', False):
            list_augmentations = []
            if self.config['dataset']['augment_gaussian']:
                list_augmentations.append(Gaussian_Noise(chance = 0.5))

            composer = Composer(transforms = list_augmentations)


        if self.config['model']['type'] == 'deeplearning':
            # Create the pytorch datasets classes
            self.train_dataset = BaseDataset(self.x_train if self.x_train_scaled is None else self.x_train_scaled, 
                                            self.y_train, 
                                            composer=composer)
            self.val_dataset = BaseDataset(self.x_val if self.x_val_scaled is None else self.x_val_scaled, 
                                           self.y_val)
            self.test_dataset = BaseDataset(self.x_test if self.x_test_scaled is None else self.x_test_scaled, 
                                            self.y_test, infer=True )

            


    def build_model(self):
        # Build the model architecture
        if self.model_type == 'deeplearning':
            # Load the model from the models folder
            self.model = NeuralNetworkTrainer(model_name=self.model_name, 
                                              config=self.config["model"], 
                                              num_classes=self.nb_classes, 
                                              train_dataset=self.train_dataset, 
                                              val_dataset=self.val_dataset, 
                                              save_path=self.experiment_path)
        else:
            # Load the model from the classical_ml folder
            self.model = ML_Wrapper(self.config['model']['name'], save_path=self.experiment_path,**self.config['model'].get('kwargs', {}))

            

    def train_model(self):
        mlflow.set_experiment(self.config["experiment_name"])
        experiment = mlflow.get_experiment_by_name(self.config["experiment_name"])
        experiment_id = experiment.experiment_id 

        with mlflow.start_run(experiment_id=experiment_id):
            self.__init_mlflow()
            # Train the model using the preprocessed data
            if self.model_type == 'deeplearning':
                # Train the deep learning model
                self.model.train()
            else:
                # Train the classical machine learning model
                self.model.train(self.x_train, self.y_train)

            # Evaluate the trained model
            self.metric_results = self.evaluate_model()
            self.interpretation_summary = self.explain_interpret()

    def explain_interpret(self):
        interpretation_summary = {}
        # Assess the robustness of the trained model
        if self.model_type == 'deeplearning':
            interpretation_summary['gaussian'] = self.model.robustness_with_captum(self.test_dataset, perturbations_mode="gaussian")
            interpretation_summary['shap_values'], _, _, _= self.model.interpretation_with_shapelet(self.test_dataset)
            interpretation_summary['list_attributions_ig'], _, _ =self.model.interpreting_with_captum(self.test_dataset, attribution_mode="Integrated Gradients")
            interpretation_summary['list_attributions_gs'], _, _ = self.model.interpreting_with_captum(self.test_dataset, attribution_mode="GradientShap")
            interpretation_summary['list_attributions_dl'], _, _ =self.model.interpreting_with_captum(self.test_dataset, attribution_mode="DeepLift")

            interpretation_summary['y_true'] = self.test_dataset.data_y


        return interpretation_summary

    def evaluate_model(self):
        # Evaluate the trained model
        if self.model_type == 'deeplearning':
            # Train the deep learning model
            self.y_pred = self.model.predict(self.test_dataset)
        else:
            # Train the classical machine learning model
            self.y_pred = self.model.predict(self.x_test if self.x_test_scaled is None else self.x_test_scaled)

        # Compute the accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred, average ="weighted")
        precision = precision_score(self.y_test, self.y_pred, average ="weighted")
        recall = recall_score(self.y_test, self.y_pred, average ="weighted")

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.log_table(pd.DataFrame(confusion_matrix(self.y_test, self.y_pred), columns=self.list_of_classes, index=self.list_of_classes).to_dict(), artifact_file="confusion_matrix.json")
        mlflow.log_table({
            "accuracy": [accuracy],
            "f1": [f1],
            "precision": [precision],
            "recall": [recall]
        }, artifact_file="metrics.json")

        print("Accuracy: {}".format(accuracy))
        print("f1: {}".format(f1))
        print("precision: {}".format(precision))
        print("recall: {}".format(recall))

        return accuracy, f1, precision, recall

    


    def run(self):
        self.load_data()
        self.preprocess_data()
        self.build_model()
        self.train_model()



if __name__ == '__main__':

    # Set the seed for reproducibility
    seed = 42
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    trainer = Trainer()
    trainer.run()

    del trainer
