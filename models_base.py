## This module contains code for physics-informed neural networks, motivated by the first 4 weeks of stuff in archive/models.py
## But most implementations in the model.py file is a bit too complicated now, so we start over with a different code structure

from __future__ import annotations
from sklearn.linear_model import LinearRegression as Linear
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod as virtual
import torch

# Filter all warnings
import warnings
warnings.filterwarnings('ignore')

__all__ = (
    "TrainingError",
    "Dataset",
    "Model"
)

class TrainingError(Exception):
    """Errors raised in the Models module"""
    pass

class Dataset:
    """A wrapper for Numpy arrays/torch tensors for easy manipulation of cutting, slicing, etc"""
    def __init__(self, data: NDArray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._data = torch.tensor(data).to(device).float()
        self._shape = data.shape

    def __repr__(self):
        return f"Dataset{self.shape}"

    @property
    def data(self):
        return self._data  
    
    # len is number of data
    def __len__(self):
        return len(self.data)
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def num_features(self):
        return self.into_iter().shape[1]
    
    def __add__(self, other: Dataset):
        if not isinstance(other, Dataset):
            raise TypeError(f"Does not support Dataset + {type(other)}")
        if len(self) != len(other):
            raise NotImplementedError(f"Cannot add dataset of different length ({len(self)} + {len(other)})")
        return ConcatenatedDataset(self.clone(), other.clone())
    
    def clone(self) -> Dataset:
        d = Dataset(torch.clone(self.data))
        d._shape = self.shape
        return d
    
    # __iter__ automatically flattens everything and yield it as a 1D tensor
    def into_iter(self) -> torch.Tensor:
        return self.data.reshape((len(self), -1))
    
    def wrap(self, shape) -> Dataset:
        """Wraps the dataset according to the given shape. Raises an error if the numbers does not match up"""
        ## Check the length
        if shape[0] != len(self):
            raise ValueError(f"The length of the shape does not match: (self: {len(self)}, shape: {shape[0]})")
        ## Do the checksum first
        checkmul = 0
        for s in shape[1:]:
            a = 1
            for k in s:
                a *= k
            checkmul += a
        orig = self.into_iter()
        num_samples = orig.shape[1]

        if checkmul != orig.shape[1]:
            raise ValueError(f"The shape does not match (self: {num_samples}, shape: {checkmul})")

        # Unwrap everything and rewrap
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i = 0
        datasets: list[Dataset] = []
        for dims in shape[1:]:
            total = 1
            for k in dims:
                total *= k
            newshape = (shape[0],) + dims
            t = torch.tensor(orig[:, i:i+total].reshape(newshape)).to(device).float()
            i += k
            d = Dataset(t)
            d._shape = newshape
            datasets.append(d)

        # Add all datasets together
        a0 = datasets[0] + datasets[1]
        for hiya in datasets[2:]:
            a0 = a0 + hiya

        return a0

    # Creates n-th degree features
    def npower(self, n: int) -> Dataset:
        PolynomialFeatures

class ConcatenatedDataset(Dataset):
    def __init__(self, data1: Dataset, data2: Dataset):
        self.data1 = data1
        self.data2 = data2
    
    @property
    def shape(self):
        if isinstance(self.data2, ConcatenatedDataset):
            return (len(self.data1), self.data1.shape[1:], *self.data2.shape[1:])
        return (len(self.data1), self.data1.shape[1:], self.data2.shape[1:])

    def clone(self) -> ConcatenatedDataset:
        return ConcatenatedDataset(self.data1.clone(), self.data2.clone())
    
    def __add__(self, other: Dataset):
        return ConcatenatedDataset(self.data1, self.data2 + other)
    
    def __len__(self):
        return len(self.data1)
    
    def into_iter(self) -> torch.Tensor:
        return torch.cat([self.data1.into_iter(), self.data2.into_iter()], axis = 1)

class Model:
    """Base class for all models. All models have a name, a fit method, and a predict method"""
    @virtual
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset):
        raise NotImplementedError
    
    @virtual
    def predict_logic(self, xtest: Dataset) -> Dataset:
        raise NotImplementedError
    
    @property
    @virtual
    def model_name(self) -> str:
        raise NotImplementedError
    
    @property
    def trained(self) -> bool:
        if not hasattr(self, "_trained"):
            return False
        return self._trained
    
    def fit(self, xtrain: Dataset, ytrain: Dataset):
        """Fit xtrain and ytrain on the model"""
        xt = xtrain.into_iter()
        yt = ytrain.into_iter()
        if xt.shape[1] > self.max_num_features:
            raise TrainingError("Too many features")
        self.fit(xt, yt)
        self._ytrain_shape = ytrain.shape
        self._trained = True
    
    def predict(self, xtest: Dataset) -> Dataset:
        """Use xtest to predict ytest"""
        if not self.trained:
            raise TrainingError("Model has not been trained")
        ypred = self.predict_logic(xtest).wrap(self._ytrain_shape)
        return ypred

    @property
    def max_num_features(self):
        """The max number of features of the model. This is useful when we want to constrain the number of parameters"""
        return 9999

# Tests
def test():
    a = np.arange(16).reshape(4, 4)
    b = np.arange(12).reshape(4, 3)

    da = Dataset(a)
    db = Dataset(b)

    dc = da + db

    assert da.shape == (4, 4)
    assert db.shape == (4, 3)
    assert dc.shape == (4, (4,), (3,))
    assert dc.into_iter().shape == (4, 7)

    d = np.arange(4199).reshape((13, 17, 19))
    dd = Dataset(d)

    assert dd.shape == (13, 17, 19)
    assert dd.into_iter().shape == (13, 17 * 19)

    try:
        da + dd
        raise AssertionError("Should not be able to add dataset of different length")
    except:
        pass

    de = Dataset(np.arange(65).reshape(13, 5))
    df = Dataset(np.arange(39).reshape(13, 3))
    dg = dd + de + df

    assert dg.shape == (13, (17, 19), (5,), (3,))
    assert dg.into_iter().shape == (13, 17 * 19 + 5 + 3)

    # Since 17 * 19 + 5 + 3 = 331 technically we can rewrap this
    dh = dg.wrap((13, (30, 11), (1,)))
    assert dh.shape == (13, (30, 11), (1,))

    try:
        dg.wrap((13, (1, 2, 3), (4, 5, 6)))
        raise AssertionError("Should not be able to rewrap dg")
    except ValueError:
        pass

if __name__ == "__main__":
    test()