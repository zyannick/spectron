import numpy as np
import torch
from models.neural_network.backbones.cnn1d import CNN1D
from models.neural_network.backbones.fcn1d import FCN_1D
from data_factory.data_reader import BaseDataset
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import time
import os

from termcolor import colored

from utils.loss_functions import FocalLoss

from captum.robust import FGSM
from captum.robust import PGD
import shap

from captum.attr import IntegratedGradients, DeepLift, NoiseTunnel, visualization, GradientShap

from utils.data_augmentations import Composer, Gaussian_Noise, Poisson_Noise, Wavelength_Shift

from sklearn.exceptions import UndefinedMetricWarning

import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class NeuralNetworkTrainer(object):
    def __init__(self, model_name : str, config : dict, num_classes : int, train_dataset : BaseDataset, val_dataset : BaseDataset, save_path : str, **kwargs):
        """Constructor for the NeuralNetworkTrainer class"""
        self.model_name = model_name
        self.save_path = save_path
        self.num_classes = num_classes
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_metrics_functions = {"accuracy": accuracy_score, "precision": precision_score, "recall": recall_score, "f1": f1_score}
        self.val_metrics_functions = {"accuracy": accuracy_score, "precision": precision_score, "recall": recall_score, "f1": f1_score}
        self.time_str = time.strftime("%Y%m%d%H%M")
        self._create_checkpoints_directorie()

    @property
    def batch_size(self):
        """Returns the batch size"""
        return self.config['batch_size']

    @property
    def num_epochs(self):
        """Returns the number of epochs"""
        return self.config['num_epochs']

    @property
    def learning_rate(self):
        """Returns the learning rate"""
        return self.config['learning_rate']

    @property
    def weight_decay(self):
        """Returns the weight decay"""
        return self.config['weight_decay']

    @property
    def optimizer(self):
        """Returns the optimizer"""
        return self.config['optimizer']

    @property
    def criterion(self):
        """Returns the criterion"""
        return self.config['criterion']

    @property
    def scheduler(self):
        """Returns the scheduler"""
        return self.config['scheduler']

    @property
    def device(self):
        """Returns the device"""
        return self.config['device']

    @property
    def num_workers(self):
        """Returns the number of workers"""
        return self.config['num_workers']

    @property
    def patience(self):
        """Returns the patience"""
        return self.config['patience']
    
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

    def _checkpointing(self):
        """
        Save the best model
        """
        # saving the compiled model

        # saving model using torchscript
        # saving the compiled model. This is useful for production
        x = torch.rand(1, 1, 800)
        model_scripted = torch.jit.trace(
            self.model, x.cuda()
        )  # Export to TorchScript
        model_scripted.save(os.path.join(self.checkdir, "scripted_model.pt"))  # Save

        # saving model using state_dict
        # saving the uncompiled model. This is useful if
        # - we want to continue training the model
        # - we want to load the model in a different architecture
        # - we want to interpret the model
        ckpt = {
            "model": self.model.state_dict(),  # type: ignore
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optimizer.state_dict(),  # type: ignore
            "scheduler": self.model_scheduler.state_dict(),  # type: ignore
            "number_of_classes": self.num_classes,
        }


        torch.save(ckpt, os.path.join(self.checkdir, "model.pt"))

        return



    def data_loader(self, dataset : BaseDataset, shuffle : bool = False):
        """Returns a data loader for the given dataset"""
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def data_loaders(self):
        """Returns the data loaders for the train and validation datasets"""
        train_loader = self.data_loader(self.train_dataset, shuffle=True)
        if len(self.val_dataset) == 0:
            return train_loader, None
        val_loader = self.data_loader(self.val_dataset, shuffle=False)
        return train_loader, val_loader


    def create_model(self):
        """Creates the model based on the model_name attribute"""
        if self.model_name == 'CNN1D':
            model = CNN1D(input_size=1, num_classes=self.num_classes)
        elif self.model_name == 'FCN_1D':
            model = FCN_1D(input_size=800, hidden_size=128, output_size=self.num_classes)
        else:
            raise NotImplementedError("Model not implemented")
        return model

    def configure_device(self):
        """Configures the device to be used for training"""
        if self.device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
        return device

    def configure_optimizers(self, model):
        """Configures the optimizer to be used for training"""
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'asgd':
            optimizer = torch.optim.ASGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("Optimizer not implemented")
        return optimizer

    def configure_criterion(self):
        """Configures the criterion to be used for training"""
        if self.criterion == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss()
        elif self.criterion == 'mse':
            criterion = torch.nn.MSELoss()
        elif self.criterion == 'l1':
            criterion = torch.nn.L1Loss()
        elif self.criterion == 'focal_loss':
            criterion = FocalLoss(gamma=2., reduction='mean')
        else:
            raise NotImplementedError("Criterion not implemented")
        return criterion


    def configure_scheduler(self, optimizer):
        """Configures the scheduler to be used for training"""
        if self.scheduler == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        else:
            raise NotImplementedError("Scheduler not implemented")
        return scheduler

    def configure_model(self, model):
        """Configures the model, optimizer, criterion and scheduler to be used for training"""
        device = self.configure_device()
        model.to(device)
        optimizer = self.configure_optimizers(model)
        criterion = self.configure_criterion()
        scheduler = self.configure_scheduler(optimizer)
        return model, optimizer, criterion, scheduler

    def val_epoch(self, model, criterion, val_loader):
        """Validates the model for one epoch"""
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                predictions = model(data)
                loss = criterion(predictions, labels)
        return loss
    


    def train_epoch(self, train_loader, val_loader):
        """Trains the model for one epoch"""
        self.model.train()
        epoch_loss = []
        self.epoch_metrics = {metric: 0.0 for metric in self.train_metrics_functions.keys()}
        epoch_labels = []
        epoch_preds = []
        with tqdm(
            total=self.steps_by_epoch,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
            unit="batch",
            position=0,
            leave=True,
            ncols=150,
        ) as pbar:
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                self.model_optimizer.zero_grad()
                logits = self.model(data)
                probabilities = torch.softmax(logits, dim=1)

                # Compute metrics
                preds = logits.detach().cpu().numpy().argmax(axis=1)

                if len(epoch_labels) > 1:
                    epoch_labels = np.concatenate((epoch_labels, labels.detach().cpu().numpy()), axis=0)
                    epoch_preds = np.concatenate((epoch_preds, preds), axis=0)
                    self.epoch_metrics["accuracy"] = self.train_metrics_functions["accuracy"](epoch_labels, epoch_preds)
                    self.epoch_metrics["precision"] = self.train_metrics_functions["precision"](epoch_labels, epoch_preds, average='macro')
                    self.epoch_metrics["recall"] = self.train_metrics_functions["recall"](epoch_labels, epoch_preds, average='macro')
                    self.epoch_metrics["f1"] = self.train_metrics_functions["f1"](epoch_labels, epoch_preds, average='macro')
                else:
                    epoch_labels = labels.detach().cpu().numpy()
                    epoch_preds = preds
                
                loss = self.model_criterion(logits, labels)
                # print(loss.item())
                loss.backward()
                self.model_optimizer.step()

                pbar.update(1)
                epoch_loss.append(loss.item())
                self.epoch_metrics["loss"] = sum(epoch_loss) / len(epoch_loss)
                for metric in self.epoch_metrics.keys():
                    if metric not in self.training_metrics.keys():
                        self.training_metrics[metric] = []
                    self.training_metrics[metric].append(self.epoch_metrics[metric])

                pbar.set_postfix(**self.epoch_metrics)


        # Validate the model    
        if val_loader is not None:
            val_loss = self.val_epoch(self.model, self.criterion, val_loader)
            print(f"Validation loss: {val_loss}")
            self.model_scheduler.step( )
            if self.best_monitor_value > val_loss:
                self.best_monitor_value = val_loss
                self.best_epoch = self.current_epoch
                print(colored(f"Best model found at epoch {self.best_epoch} and loss {val_loss}", "green"))
                self._checkpointing()
            else:
                print(colored(f"Best model found at epoch {self.best_epoch} and loss {val_loss}", "red"))
        else:
            self.model_scheduler.step( )
            if self.best_monitor_value > np.mean(epoch_loss):
                self.best_monitor_value = np.mean(epoch_loss)
                self.best_epoch = self.current_epoch
                print(colored(f"Best model found at epoch {self.best_epoch} and loss {np.mean(epoch_loss)}", "green"))
                self._checkpointing()
            else:
                print(colored(f"Best model found at epoch {self.best_epoch} and loss {np.mean(epoch_loss)}", "red"))

        self.current_lr = self.model_scheduler.get_last_lr()
        
        print('Learning rate: {}'.format(self.current_lr))

        continue_training = self.current_epoch - self.best_epoch < self.patience

        return continue_training
    
    def predict(self, dataset : BaseDataset):
        """Predicts the labels for the given data loader"""
        if torch.cuda.is_available():
            self.model = torch.jit.load(os.path.join(self.checkdir, "scripted_model.pt"), map_location=torch.device("cuda"))
        else:
            self.model = torch.jit.load(os.path.join(self.checkdir, "scripted_model.pt"), map_location=torch.device("cpu"))

        self.model.to(self.device)
        self.model.eval()
        print("Evaluation of the train model")
        list_of_predictions = []
        with torch.no_grad():
            for index in tqdm(range(len(dataset)), total=len(dataset)):
                data, _ = dataset.__getitem__(index)
                data = data.to(self.device)
                predictions = self.model(data)
                list_of_predictions.append(predictions.detach().cpu().numpy().argmax(axis=1)[0])
        return list_of_predictions
    
    
    def robustness_with_captum(self, dataset : BaseDataset, perturbations_mode : str = "gaussian"):
        """Test the robutness of the model with regards to fgsm perturbations"""
        if torch.cuda.is_available():
            self.model = torch.jit.load(os.path.join(self.checkdir, "scripted_model.pt"), map_location=torch.device("cuda"))
        else:
            self.model = torch.jit.load(os.path.join(self.checkdir, "scripted_model.pt"), map_location=torch.device("cpu"))

    
        if perturbations_mode == 'gaussian':
            gaussian = Gaussian_Noise(chance=1)
            pertubation_model = gaussian
            print("Gaussian perturbations")
        elif perturbations_mode == 'poisson':
            poisson = Poisson_Noise(chance=1)
            pertubation_model = poisson
            print("Poisson perturbations")

        data = dataset.data_x
        labels = dataset.data_y


        self.model.to(self.device)
        self.model.eval()
        print("Evaluationthe robustness of the train model")
        list_of_predictions = []
        
        with torch.no_grad():
            for ind, (x, y) in enumerate(zip(data, labels)):
                x = pertubation_model(x) 
                x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
                predictions = self.model(x)
                list_of_predictions.append(predictions.detach().cpu().numpy().argmax(axis=1)[0])

        return list_of_predictions
    
    def interpretation_with_shapelet(self, dataset : BaseDataset):
        """interpret the model with shapelet"""
        self.model = self.create_model()
        if torch.cuda.is_available():
            ckpt_best = torch.load(os.path.join(self.checkdir, "model.pt"), map_location=torch.device("cuda"))
        else:
            ckpt_best = torch.load(os.path.join(self.checkdir, "model.pt"), map_location=torch.device("cpu"))

        self.model.load_state_dict(ckpt_best["model"])

        self.model.to(self.device)
        self.model.eval()
        print("Evaluation of the train model")
        list_of_predictions = []

        test_data = dataset.data_x
        test_labels = dataset.data_y
        background = torch.from_numpy( test_data[np.random.choice(test_data.shape[0], 100, replace=False)].astype(np.float32) ).to(self.device)
        e = shap.DeepExplainer(self.model, background)
        shap_values = e.shap_values( torch.from_numpy(test_data.astype(np.float32) ).to(self.device) , check_additivity=False)

        return shap_values, background, test_data, test_labels
    
    def interpreting_with_captum(self, dataset : BaseDataset, attribution_mode : str = "Integrated Gradients"):
        """interpret the model with captum"""
        self.model = self.create_model()
        if torch.cuda.is_available():
            ckpt_best = torch.load(os.path.join(self.checkdir, "model.pt"), map_location=torch.device("cuda"))
        else:
            ckpt_best = torch.load(os.path.join(self.checkdir, "model.pt"), map_location=torch.device("cpu"))

        self.model.load_state_dict(ckpt_best["model"])

        self.model.to(self.device)

        data = dataset.data_x
        labels = dataset.data_y

        def attribute_spectra_features(algorithm, input, ind, **kwargs):
            self.model.zero_grad()
            tensor_attributions = algorithm.attribute(input,
                                                    target=labels[ind],
                                                    **kwargs
                                                    )
            
            return tensor_attributions
        
        list_attributions = []
        
        for ind, (x, y) in enumerate(zip(data, labels)):
            input = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            input.requires_grad = True
            if attribution_mode == 'Integrated Gradients':
                ig = IntegratedGradients(self.model)
                nt = NoiseTunnel(ig)
                attr_ig_nt = attribute_spectra_features(nt, input, ind , baselines=input * 0, nt_type='smoothgrad_sq',
                                                    nt_samples=100, stdevs=0.2)
                attr_ig_nt = attr_ig_nt.squeeze(0).cpu().detach().numpy()
                list_attributions.append(attr_ig_nt)

            elif attribution_mode == 'DeepLift':
                dl = DeepLift(self.model)
                nt = NoiseTunnel(dl)
                attr_dl_nt = attribute_spectra_features(nt, input, ind, baselines=input * 0, nt_type='smoothgrad_sq',
                                                    nt_samples=100, stdevs=0.2)
                attr_dl_nt = attr_dl_nt.squeeze(0).cpu().detach().numpy()
                list_attributions.append(attr_dl_nt)

            elif attribution_mode == 'GradientShap':
                gs = GradientShap(self.model)
                attr_gs = attribute_spectra_features(gs, input,ind, baselines=input * 0, n_samples=10)
                attr_gs = attr_gs.squeeze(0).cpu().detach().numpy()
                list_attributions.append(attr_gs)

        return list_attributions, data, labels
        


    def train(self):
        self.model, self.model_optimizer, self.model_criterion, self.model_scheduler = self.configure_model(self.create_model())
        train_loader, val_loader = self.data_loaders()
        self.steps_by_epoch = len(self.train_dataset) // self.batch_size

        self.training_metrics = {}

        # Set some parameters to get the best model
        self.monitor_value = np.inf
        self.best_epoch = 0
        self.best_monitor_value = np.inf

        # Loop over the epochs
        for self.current_epoch in range(self.num_epochs):
            # Train one epoch
            continue_training = self.train_epoch(train_loader, val_loader)
            # Print progress
            if not continue_training:
               break

        for metric in self.training_metrics.keys():
            plt.figure()
            plt.plot(self.training_metrics[metric], label=metric)
            plt.legend()
            plt.savefig(os.path.join(self.checkdir, metric + ".png"))

        del self.model,  self.model_optimizer, self.model_criterion, self.model_scheduler