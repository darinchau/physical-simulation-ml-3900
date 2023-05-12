## This module contains code for physics-informed neural networks, motivated by the first 4 weeks of stuff in archive/models.py
## But most implementations in the model.py file is a bit too complicated now, so we start over with a different code structure

from __future__ import annotations
import os
from typing import Any
from sklearn.linear_model import LinearRegression, RidgeCV, ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod as virtual
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from models_base import *
from torch import Tensor
from load import load_spacing
from derivative import PoissonLoss
from models_base import Dataset

__all__ = (
    "BayesianRegressionModel",
    "Dataset",
    "ElectronDensityInformedModel",
    "GaussianModel",
    "LinearAugmentedLSTMModel",
    "LinearAugmentedModel",
    "LinearLSTMModel",
    "LinearModel",
    "Model",
    "ModelFactory",
    "PoissonModel",
    "RidgeAugmentedModel",
    "RidgeModel",
    "SpaceChargeInformedModel",
    "StochasticLSTMModel",
    "SymmetricPoissonModel",
    "TrainingError",
)

## For example this is how you can wrap around a linear model
class LinearModel(Model):
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        xt = xtrain.to_tensor().cpu().numpy()
        yt = ytrain.to_tensor().cpu().numpy()
        return LinearRegression().fit(xt, yt)
    
    def predict_logic(self, model: LinearRegression, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor().cpu().numpy()
        ypred = model.predict(xt)
        return Dataset(torch.as_tensor(ypred))
    
## For example this is how you can wrap around a ridge model
class RidgeModel(Model):
    """Uses the Ridge Cross Validation model"""
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        xt = xtrain.to_tensor().cpu().numpy()
        yt = ytrain.to_tensor().cpu().numpy()
        return RidgeCV().fit(xt, yt)
    
    def predict_logic(self, model: LinearRegression, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor().cpu().numpy()
        ypred = model.predict(xt)
        return Dataset(torch.as_tensor(ypred))

## Here is how you can wrap around the Gaussian prediction
class GaussianModel(Model):
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        kernel = RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)
        xt = xtrain.to_tensor().cpu().numpy()
        yt = ytrain.to_tensor().cpu().numpy()
        return model.fit(xt, yt)
    
    def predict_logic(self, model: GaussianProcessRegressor, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor().cpu().numpy()
        ypred = model.predict(xt)
        return Dataset(torch.as_tensor(ypred))
    
# The augmented models
class LinearAugmentedModel(AugmentedModel, LinearModel):
    pass
    
class RidgeAugmentedModel(AugmentedModel, RidgeModel):
    pass

class ElectronDensityInformedModel(Model):
    """Two-layer informed model by electron density. This describes the linear-linear model
    where we try to use a two-layer model to first predict e density and use e density 
    and x to predict the other data"""
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        # Heuristically edensity only works the same as edensity + space charge
        edensity = self.informed["edensity"]
        xt = xtrain + xtrain.square() + xtrain.nexp()
        model_x_ed = LinearModel().fit(xt, edensity)
        model_xed_y = LinearModel().fit(xtrain + edensity, ytrain)
        return model_x_ed, model_xed_y
    
    def predict_logic(self, model: tuple[Model, Model], xtest: Dataset) -> Dataset:
        model_x_ed, model_xed_y = model
        xt = xtest + xtest.square() + xtest.nexp()
        edensity = model_x_ed.predict(xt)
        ypred = model_xed_y.predict(xtest + edensity)
        return ypred
    
class SpaceChargeInformedModel(ElectronDensityInformedModel):
    """Two-layer informed model by space charge. This describes the linear-linear model
    where we try to use a two-layer model to first predict e density and use e density 
    and x to predict the other data"""
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        # Heuristically edensity only works the same as edensity + space charge
        edensity = self.informed["spacecharge"]
        xt = xtrain + xtrain.square() + xtrain.nexp()
        model_x_ed = LinearModel().fit(xt, edensity)
        model_xed_y = LinearModel().fit(xtrain + edensity, ytrain)
        return model_x_ed, model_xed_y

class PoissonNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 100),
            nn.Sigmoid(),
            nn.Linear(100, 2193),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Possion equation verifier - using autograd to try and verify neural nets
class PoissonModel(NeuralNetModel):
    """An implementation of the physics informed neural networks (PINN) following the ideas in https://arxiv.org/pdf/1711.10561.pdf
    We train a neural network while simultaneously trying to verify that the neural network as a function
    satisfies the poisson equation using the informed data."""
    def neural_net(self) -> nn.Module:
        # This factorization exists to make Symmetric Poisson implmemntation easier
        return PoissonNN()
    
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset, epochs: int, verbose: bool = True) -> Any:
        device = self._device
        xtest = self.informed["xtest"].to_tensor().to(device)
        ytest = self.informed["ytest"].to_tensor().to(device)
        sctrain = self.informed["spacecharge"].to_tensor().to(device)
        sctest = self.informed["spacecharge-test"].to_tensor().to(device)
        path: str = self.informed["path"]

        xtrain = xtrain.to_tensor().to(device)
        ytrain = ytrain.to_tensor().to(device)

        net = self.neural_net().to(device).double()
        optimizer = optim.LBFGS(net.parameters(), lr=0.01)
        criterion1 = nn.MSELoss()
        criterion2 = PoissonLoss()

        history = History()
        self._logs = []

        # Flag the model as dangerous if epochs too big
        dangerous = 0

        # Batch size is 1 for us so better use this approach instead
        for epoch in range(epochs):
            # Update the history
            history.update()

            # Train loop
            train_mse, train_poi = 0., 0.
            for (x, y, sc) in zip(xtrain, ytrain, sctrain):
                def closure():
                    nonlocal train_mse
                    nonlocal train_poi
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    ypred = net(x)
                    mse = criterion1(ypred, y)
                    poissonloss = criterion2(ypred, sc)
                    loss = mse + poissonloss
                    if loss.requires_grad:
                        loss.backward()
                    train_mse += mse.item()
                    train_poi += poissonloss.item()
                    return loss
                optimizer.step(closure)
            
            # Test loop
            test_mse, test_poi = 0., 0.
            with torch.no_grad():
                for (x, y, sc) in zip(xtest, ytest, sctest):
                    ypred = net(x)
                    mse = criterion1(ypred, y)
                    poissonloss = criterion2(ypred, sc)
                    loss = mse + poissonloss
                    test_mse += mse.item()
                    test_poi += poissonloss.item()

            # Abort early if we found our model exploding
            train_loss = train_mse + train_poi
            test_loss = test_poi + test_mse                

            # Print logs if necessary
            log = f"Trained {epoch + 1} epochs on {self.name} for {self.informed['inputname']} with train loss {train_loss}, test_loss {test_loss}"
            self._logs.append(log)
            if verbose:
                print(log)

            # Append the history
            history.train(float(train_mse)/len(xtrain), "MSE loss")
            history.train(float(train_poi)/len(xtrain), "Poisson loss")
            history.test(float(test_mse)/len(xtest), "MSE loss")
            history.test(float(test_poi)/len(xtest), "Poisson loss")

            # Save model and abort early if necessary
            if epoch > 3 and (train_loss > 1000 or test_loss > 1000):
                dangerous += 1
            else:
                dangerous = 0
            
            if dangerous >= 3:
                self._logs.append(f"Aborting early at epoch {epoch + 1}")
                break

            if dangerous == 0:
                self._logs.append(f"Saving model at epoch {epoch + 1}")
                torch.save(net.state_dict(), f"{path}/{self.informed['inputname']}.pth")

        # Load the best dict
        net = self.neural_net()
        net.load_state_dict(torch.load(f"{path}/{self.informed['inputname']}.pth"))
        net = net.to(device).double()

        # Save one's history somewhere
        return net, history

class SymmetricNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 100),
            nn.Sigmoid(),
            nn.Linear(100, 17 * 72),
            nn.Sigmoid()
        )
        self.x_spacing = load_spacing()[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.reshape(-1, 72, 17)
        total = 0.078
        
        target = torch.zeros(x.shape[0], 129, 17).to(self.device).double()
        target[:, :72, :] = x

        for j in range(72, 129):
            x_pos = total - self.x_spacing[j]

            # Use normal flip near the edges
            if total - x_pos < 0.02:
                col = torch.abs(x_pos - self.x_spacing).argmin()
                target[:, j, :] = x[:, col, :]
                continue
            
            # Use lerp near the center
            col1 = torch.searchsorted(self.x_spacing, x_pos, side='right') - 1
            col2 = col1 + 1
            weighting = (self.x_spacing[col2] - x_pos)/(self.x_spacing[col2] - self.x_spacing[col1])
            target[:, j, :] = weighting * x[:, col1, :] + (1-weighting) * x[:, col2, :]
        
        return target.reshape(-1, 2193)
    
# Mix symmetric and Poisson
class SymmetricPoissonModel(PoissonModel):
    """PoissonNN, but using the symmetric model"""
    def neural_net(self) -> nn.Module:
        return SymmetricNN() 

# This is a debug model for the Time series model
class LinearTimeSeriesModel(TimeSeriesModel):
    """A debug model"""
    def __init__(self, i: int = 1):
        super().__init__()
        self.i = i
    
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        return LinearModel().fit(xtrain, ytrain)
    
    def predict_logic(self, model, xtest: Dataset) -> Dataset:
        return model.predict(xtest)

# Since we can replicate the poisson loss, can we try to make an initial prediction and 
# nudge the result until it satisfies the poisson equation?
# This serves to be useful in a future model
class LinearLSTMModel(TimeSeriesModel):
    """Use Bayesian logic to predict outcome based on past N results"""
    def get_model(self):
        return RidgeModel()
    
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        N = self.N
        # First use a Bayesian model to predict the next data based on the prev 4 data
        # Simultaneously damp the xtrain past values so that nothing goes out of hand
        last_N: Tensor = torch.stack([xtrain.datas[i+1] for i in range(N)], axis = -1)
        last_N = last_N.reshape(-1, N)
        last_N = last_N.double()
        
        # Add back the voltage as the argument
        vgs = xtrain.datas[0].reshape(-1, 1, 1) + torch.zeros(1, 129, 17)
        vgs = vgs.reshape(-1, 1)

        self._yshape = ytrain.datas[0].shape

        new_x = Dataset(vgs, last_N)
        new_y = Dataset(ytrain.datas[0].reshape(-1, 1))

        return self.get_model().fit(new_x, new_y)
    
    def predict_logic(self, model: Model, xtest: Dataset) -> Dataset:
        N = self.N
        # First use a Bayesian model to predict the next data based on the prev 4 data
        last_N: Tensor = torch.stack([xtest.datas[i+1] for i in range(N)], axis = -1)
        last_N = last_N.reshape(-1, N)
        last_N = last_N.double()
        
        # Add back the voltage as the argument
        vgs = xtest.datas[0].reshape(-1, 1, 1) + torch.zeros(1, 129, 17)
        vgs = vgs.reshape(-1, 1)

        yshape = (len(xtest),) + self._yshape[1:]

        new_x = Dataset(vgs, last_N)
        ypred = model.predict(new_x)
        ypred = Dataset(ypred.datas[0].view(yshape))
        return ypred

# Variations on the same theme above
class LinearAugmentedLSTMModel(LinearLSTMModel):
    def get_model(self):
        return RidgeAugmentedModel()

class BayesianRegressionModel(MultiModel):
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        xt = xtrain.to_tensor().cpu().numpy()
        yt = ytrain.to_tensor().cpu().numpy()
        return ARDRegression().fit(xt, yt)
    
    def predict_logic(self, model: ARDRegression, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor().cpu().numpy()
        ypred = model.predict(xt).reshape((-1, 1))
        return Dataset(torch.as_tensor(ypred))

class StochasticLSTMModel(LinearLSTMModel):
    def get_model(self):
        return BayesianRegressionModel()

class PoissonAutoRegressionModel:
    """This model takes in last n data, last n space charge, and the vg, and predicts this data and this space charge"""
    