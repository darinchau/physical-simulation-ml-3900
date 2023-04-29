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
    "TLILinearLinearModel",
    "TLIGaussianLinearModel",
    "PoissonNNModel"
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

class TLIModel(Model):
    """Two-layer informed model by electron density. This describes a general class of models 
    where we try to use a two-layer model to first predict e density and use e density 
    and x to predict the other data"""
    @property
    @virtual
    def model_1(self) -> Model:
        raise NotImplementedError

    @property
    @virtual
    def model_2(self) -> Model:
        raise NotImplementedError
    
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

class TLILinearLinearModel(TLIModel):
    @property
    def model_1(self):
        return LinearModel()
    
    @property
    def model_2(self):
        return LinearModel()

class TLIGaussianLinearModel(TLIModel):
    @property
    def model_1(self) -> Model:
        return GaussianModel()
    
    @property
    def model_2(self) -> Model:
        return LinearModel()
    
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
    def __init__(self):
        super(PoissonLoss, self).__init__()
        self.x, self.y = load_spacing()

    def forward(self, output: Tensor, target: Tensor, space_charge: Tensor) -> Tensor:
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        dPdx, dPdy = torch.gradient(output, spacing = (self.x, self.y))
        d2Pdx2 = torch.gradient(dPdx, spacing = (self.x, self.y))[0]
        d2Pdy2 = torch.gradient(dPdy, spacing = (self.x, self.y))[1]
        possion_loss = d2Pdx2 + d2Pdy2 - space_charge
        return loss + possion_loss

# Possion equation verifier - using autograd to try and verify neural nets
class PoissonNNModel(Model):
    """An implementation of the physics informed neural networks (PINN) following the ideas in https://arxiv.org/pdf/1711.10561.pdf
    We train a neural network while simultaneously trying to verify that the neural network as a function
    satisfies the poisson equation using the informed data."""
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        xtest, ytest = self.informed["xtest"], self.informed["ytest"]
        sctrain, sctest = self.informed["spacecharge"], self.informed["spacecharge-test"]
        xtrain, ytrain = xtrain.to_tensor().to(device), ytrain.to_tensor().to(device)

        net = PoissonNN().to(device)
        optimizer = optim.LBFGS(net.parameters(), lr=0.01)
        criterion = PoissonLoss()

        history = History()

        epochs = 3000

        # Batch size is 1 for us so better use this approach instead
        for i in range(epochs):
            # Train loop
            train_loss = 0.
            for (x, y, sc) in zip(xtrain, ytrain, sctrain):
                optimizer.zero_grad()
                ypred = net(x)
                loss = criterion(ypred, y, sc)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            history.train(train_loss)

            # Only test every 10 times
            if i % 10 < 9:
                continue
            
            # Test loop
            test_loss = 0.
            with torch.no_grad():
                for (x, y, sc) in zip(xtest, ytest, sctest):
                    output = net(x)
                    loss = criterion(ypred, y, sc)
                    test_loss += loss.item()
            history.test(test_loss)
        
        # Save one's history somewhere
        self._history = history
        self._net = net
        return net
    
    def predict_logic(self, model: nn.Module, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor()
        output = model(xt)
        return Dataset(output)
    
    def save_model(self, root: str, name: str):
        self._history.plot(root, name)
        model_scripted = torch.jit.script(self._net)
        model_scripted.save(f'{root}/{name}.pt')
