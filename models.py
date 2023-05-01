## This module contains code for physics-informed neural networks, motivated by the first 4 weeks of stuff in archive/models.py
## But most implementations in the model.py file is a bit too complicated now, so we start over with a different code structure

from __future__ import annotations
from typing import Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod as virtual
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from models_base import TrainingError, Model, Dataset, History
from torch import Tensor
from load import load_spacing

__all__ = (
    "TrainingError",
    "Model",
    "Dataset",
    "LinearModel",
    "GaussianModel",
    "LinearLinearInformedModel",
    "PoissonNNModel",
    ""
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

class LinearLinearInformedModel(Model):
    """Two-layer informed model by electron density. This describes the linear-linear model
    where we try to use a two-layer model to first predict e density and use e density 
    and x to predict the other data"""
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        # Heuristically edensity only works the same as edensity + space charge
        edensity = self.informed["edensity"]
        xt = xtrain + xtrain.square() + xtrain.exp()
        model_x_ed = self.model_1.fit(xt, edensity)
        model_xed_y = self.model_2.fit(xtrain + edensity, ytrain)
        return model_x_ed, model_xed_y
    
    def predict_logic(self, model: tuple[Model, Model], xtest: Dataset) -> Dataset:
        model_x_ed, model_xed_y = model
        xt = xtest + xtest.square() + xtest.exp()
        edensity = model_x_ed.predict(xt)
        ypred = model_xed_y.predict(xtest + edensity)
        return ypred
    
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
    
class PoissonLoss(nn.Module):
    """Gives the poisson equation - the value of ∇²φ = S
    where S is the space charge described in p265 of the PDF 
    https://www.researchgate.net/profile/Nabil-Ashraf/post/How-to-control-the-slope-of-output-characteristicsId-Vd-of-a-GAA-nanowire-FET-which-shows-flat-saturated-region/attachment/5de3c15bcfe4a777d4f64432/AS%3A831293646458882%401575207258619/download/Synopsis_Sentaurus_user_manual.pdf"""
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

# Possion equation verifier - using autograd to try and verify neural nets
class PoissonNNModel(Model):
    """An implementation of the physics informed neural networks (PINN) following the ideas in https://arxiv.org/pdf/1711.10561.pdf
    We train a neural network while simultaneously trying to verify that the neural network as a function
    satisfies the poisson equation using the informed data."""
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset, epochs: int = 3000, verbose: bool = True) -> Any:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        xtest = self.informed["xtest"].to_tensor().to(device)
        ytest = self.informed["ytest"].to_tensor().to(device)
        sctrain = self.informed["spacecharge"].to_tensor().to(device)
        sctest = self.informed["spacecharge-test"].to_tensor().to(device)
        xtrain = xtrain.to_tensor().to(device)
        ytrain = ytrain.to_tensor().to(device)

        net = PoissonNN().to(device)
        optimizer = optim.LBFGS(net.parameters(), lr=0.01)
        criterion1 = nn.MSELoss()
        criterion2 = PoissonLoss()

        history = History()
        self._logs = []

        # Batch size is 1 for us so better use this approach instead
        for i in range(epochs):
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

            history.train(float(train_mse)/len(xtrain), "MSE loss")
            history.train(float(train_poi)/len(xtrain), "Poisson loss")

            # Only test every 10 times
            if i % 10 < 9:
                continue
            
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
            history.test(float(test_mse)/len(xtest), "MSE loss")
            history.test(float(test_poi)/len(xtest), "Poisson loss")

            # Print logs if necessary
            log = f"Trained {i} epochs with train loss {train_mse + train_poi}, test_loss {test_poi + test_mse}"
            self._logs.append(log)
            if verbose:
                print(log)
        
        # Save one's history somewhere
        self._history = history
        self._net = net
        return net
    
    def predict_logic(self, model: nn.Module, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor()
        output = model(xt)
        return Dataset(output)
    
    def save(self, root: str, name: str):
        self._history.plot(root, name)
        model_scripted = torch.jit.script(self._net)
        model_scripted.save(f'{root}/{name}.pt')

        with open(f"{root}/{name} history.txt", 'w') as f:
            for log in self._logs:
                f.write(log)

class SymetricNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 100),
            nn.Sigmoid(),
            nn.Linear(100, 17 * 69),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = x.reshape(-1, 69, 17)

class SymmetricNNModel(Model):
    """Uses the fact the thing is almost symmetric and tries to define some loss to penalise the training if the result is not symmetric"""
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset, epochs: int = 3000, verbose: bool = True) -> Any:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        xtest = self.informed["xtest"].to_tensor().to(device)
        ytest = self.informed["ytest"].to_tensor().to(device)
        sctrain = self.informed["spacecharge"].to_tensor().to(device)
        sctest = self.informed["spacecharge-test"].to_tensor().to(device)
        xtrain = xtrain.to_tensor().to(device)
        ytrain = ytrain.to_tensor().to(device)

        net = PoissonNN().to(device)
        optimizer = optim.LBFGS(net.parameters(), lr=0.01)
        criterion1 = nn.MSELoss()
        criterion2 = PoissonLoss()

        history = History()
        self._logs = []

        # Batch size is 1 for us so better use this approach instead
        for i in range(epochs):
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

            history.train(float(train_mse)/len(xtrain), "MSE loss")
            history.train(float(train_poi)/len(xtrain), "Poisson loss")

            # Only test every 10 times
            if i % 10 < 9:
                continue
            
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
            history.test(float(test_mse)/len(xtest), "MSE loss")
            history.test(float(test_poi)/len(xtest), "Poisson loss")

            # Print logs if necessary
            log = f"Trained {i} epochs with train loss {train_mse + train_poi}, test_loss {test_poi + test_mse}"
            self._logs.append(log)
            if verbose:
                print(log)
        
        # Save one's history somewhere
        self._history = history
        self._net = net
        return net
    
    def predict_logic(self, model: nn.Module, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor()
        output = model(xt)
        return Dataset(output)
    
    def save(self, root: str, name: str):
        self._history.plot(root, name)
        model_scripted = torch.jit.script(self._net)
        model_scripted.save(f'{root}/{name}.pt')

        with open(f"{root}/{name} history.txt", 'w') as f:
            for log in self._logs:
                f.write(log)
