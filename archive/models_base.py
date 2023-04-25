## This module contains code for physics-informed neural networks, motivated by the first 4 weeks of stuff in archive/models.py
## But most implementations in the model.py file is a bit too complicated now, so we start over with a different code structure

from __future__ import annotations
from sklearn.linear_model import LinearRegression as Linear
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod as virtual
from load import index_exclude, index_include
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary
from numpy import log
from scipy.optimize import minimize
from numba import njit

# Filter all warnings
import warnings
warnings.filterwarnings('ignore')

__all__ = (
    "TrainingError",
    "Dataset",
    "Model"
)

class TrainingError(Exception):
    """Errors raised in the PINN module"""
    pass

class Dataset:
    """A wrapper for Numpy arrays/torch tensors for easy manipulation of cutting, slicing, etc"""
    def __init__(self, data: torch.Tensor):
        self.data = data
        self._shape: tuple = self.data.cpu().numpy().shape
    
    @classmethod
    def from_array(cls, data: NDArray) -> Dataset:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(torch.tensor(data).to(device).float())
    
    # len is number of data
    def __len__(self):
        return len(self.data)
    
    @property
    def shape(self):
        return self._shape
    
    def __add__(self, other: Dataset):
        if len(self) != len(other):
            raise NotImplementedError(f"Cannot add dataset of different length ({len(self)} + {len(other)})")
        return ConcatenatedDataset(self.clone(), other.clone())
    
    def clone(self) -> Dataset:
        return Dataset(torch.clone(self.data))
    
    # __iter__ automatically flattens everything and yield it as a 1D tensor
    def into_iter(self):
        return self.data.reshape((len(self), -1))

class ConcatenatedDataset(Dataset):
    def __init__(self, data1: Dataset, data2: Dataset):
        self.data1 = data1
        self.data2 = data2
    
    @property
    def shape(self):
        return tuple(len(self.data1), *self.data1.shape[1:], *self.data2.shape[1:])
    
    def clone(self) -> ConcatenatedDataset:
        return ConcatenatedDataset(self.data1.clone(), self.data2.clone())
    
    def __add__(self, other: Dataset):
        return ConcatenatedDataset(self.data1, self.data2 + other)
    
    def __len__(self):
        return len(self.data1)
    
    def into_iter(self):
        return torch.cat([self.data])


class Model:
    """Base class for all models. All models have a name, a fit method, and a predict method"""
    @virtual
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        raise NotImplementedError
    
    @virtual
    def predict_logic(self, xtest: Dataset) -> Dataset:
        raise NotImplementedError
    
    @virtual
    @property
    def model_name(self) -> str:
        raise NotImplementedError
    
    @property
    def trained(self) -> bool:
        if not hasattr(self, "_trained"):
            return False
        return self._trained
    
    def fit(self, xtrain: Dataset, ytrain: Dataset):
        self.fit(xtrain, ytrain)
        self._trained = True
    
    def predict(self, xtest: Dataset) -> Dataset:
        if not self.trained:
            raise TrainingError("Model has not been trained")
        return self.predict_logic(xtest)
