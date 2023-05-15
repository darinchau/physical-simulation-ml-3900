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
from modules import get_device
from torch import Tensor
from load import *
from modules import PoissonNN, PoissonLoss, SymmetricNN
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
    "LSTMModel",
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

# Possion equation verifier - using autograd to try and verify neural nets
class PoissonModel(NeuralNetModel):
    """An implementation of the physics informed neural networks (PINN) following the ideas in https://arxiv.org/pdf/1711.10561.pdf
    We train a neural network while simultaneously trying to verify that the neural network as a function
    satisfies the poisson equation using the informed data."""
    def neural_net(self) -> nn.Module:
        # This factorization exists to make Symmetric Poisson implmemntation easier
        return PoissonNN()
    
    def fit_logic(self, xtrain_: Dataset, ytrain_: Dataset, epochs: int, verbose: bool = True) -> Any:
        device = self._device
        xtest = self.informed["xtest"].to_tensor().to(device)
        ytest = self.informed["ytest"].to_tensor().to(device)
        sctrain = self.informed["spacecharge"].to_tensor().to(device)
        sctest = self.informed["spacecharge-test"].to_tensor().to(device)
        path: str = self.informed["path"]

        xtrain = xtrain_.to_tensor().to(device)
        ytrain = ytrain_.to_tensor().to(device)

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

class LSTMModel(Model):
    """Prof. Wong and Albert's idea for the LSTM model"""
    def get_model(self) -> Model:
        return LinearModel()
    
    def get_fallback(self) -> Model:
        return LinearRegression()
    
    @property
    def min_training_data(self) -> int:
        return 2
    
    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset):
        # Check if data is one dimensional
        if xtrain.shape[1][0] != 1:
            raise ValueError("Currently, Time series model abstract class only works on one-dimensional data, which is the time")
        
        return super()._fit_inner(xtrain, ytrain)
    
    def fit_logic(self, x: Dataset, y: Dataset):
        space_charge = self.informed["spacecharge"]
        new_x = x[1:] + space_charge[:-1] + y[:-1]
        new_y = y[1:] + space_charge[1:]
        model = self.get_model().fit(new_x, new_y)
        self._x = x
        self._xsc = space_charge
        self._y = y

        xt = x.to_tensor().cpu().numpy()
        yt = y.to_tensor().cpu().numpy()
        linear = self.get_fallback().fit(xt, yt)
        return model, linear
    
    def predict_logic(self, model, xtest: Dataset) -> Dataset:
        fw, linear = model
        assert isinstance(linear, LinearRegression)
        assert isinstance(fw, Model)

        # Forecase all first
        # Calculate the time step difference
        time_step_diff = (self._x.datas[0][1] - self._x.datas[0][0])[0]
        furthest_time_step = torch.max(xtest.to_tensor())

        # Loop until we forecast enough data to return
        next_x = self._x.datas[0][-1][0]
        while next_x <= furthest_time_step:
            # Add increment
            next_x += time_step_diff

            # Save x first
            new_x = Dataset(torch.tensor([[next_x]]))
            self._x += new_x

            # Build in the last time step results
            new_x = new_x + self._xsc[-1] + self._y[-1]
            ypred, scpred = fw.predict(new_x)
            self._y += ypred
            self._xsc += scpred

        # Make predictions one by one
        predictions: list[Dataset] = []
        sorted_tensor, indices = torch.sort(self._x.datas[0][:, 0])
        sorted_tensor = sorted_tensor.reshape(-1)
        for i in range(len(xtest)):
            xi = xtest[i]
            x = xi.datas[0][0][0]

            # Case 1: smaller than everything in x
            x_min = torch.min(self._x.datas[0])
            if x < x_min:
                ypred = Dataset(linear.predict(xi.to_tensor().cpu().numpy()))
                predictions.append(ypred)
                continue

            # Case 2: in between something in x
            index_left = torch.searchsorted(sorted_tensor, x, right=False)
            index_right = torch.searchsorted(sorted_tensor, x, right=True)
            left, right = indices[index_left], indices[index_right]
            if left == right:
                ypred = Dataset(self._y.datas[0][left].reshape(1, -1))
                predictions.append(ypred)
                continue
            
            # predict the result linearly
            new_xtrain = self._x.datas[0][(left, right), :].cpu().numpy().reshape(2, -1)
            new_ytrain = self._y.datas[0][(left, right), :].cpu().numpy().reshape(2, -1)
            lin = LinearRegression().fit(new_xtrain, new_ytrain)
            ypred = Dataset(lin.predict(np.array([[x]])))
            predictions.append(ypred)
        
        # Return the predictions
        if len(predictions) == 1:
            return predictions[0]
        
        d = predictions[0]
        for i in range(1, len(predictions)):
            d += predictions[i]
        return d