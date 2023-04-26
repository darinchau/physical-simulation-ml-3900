## This module contains code for physics-informed neural networks, motivated by the first 4 weeks of stuff in archive/models.py
## But most implementations in the model.py file is a bit too complicated now, so we start over with a different code structure

from __future__ import annotations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod as virtual
from load import index_exclude, index_include
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from models_base import TrainingError, Model, Dataset

__all__ = (
    "LinearModel"
)

## For example this is how you can wrap around a linear model
class LinearModel(Model):
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        self.model = LinearRegression().fit(xtrain, ytrain)
    
    def predict_logic(self, xtest: Dataset) -> Dataset:
        return self.model.predict(xtest)
