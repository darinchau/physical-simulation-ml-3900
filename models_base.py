## This module contains code for base classes of models, motivated by the first 4 weeks of stuff in archive/models.py
## But most implementations in the train.py file is a bit too complicated now, so we start over with a different code structure

from __future__ import annotations
from sklearn.linear_model import LinearRegression as Linear
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod as virtual, ABC
import torch
from torch import Tensor, nn
from typing import Any
import matplotlib.pyplot as plt

# Filter all warnings
import warnings
warnings.filterwarnings('ignore')

__all__ = (
    "TrainingError",
    "Dataset",
    "Model",
    "MultiModel",
    "History"
)

class TrainingError(Exception):
    """Errors raised in the Models module which are recoverable and can be simply and safely skipped over."""
    pass

class Dataset:
    """A wrapper for Numpy arrays/torch tensors for easy manipulation of cutting, slicing, etc"""
    def __init__(self, *datas: NDArray | Tensor):
        # Check if the have the same first dimension
        num_entries = tuple(tuple(x.shape)[0] for x in datas)
        for i, entry in enumerate({num_entries[0]}):
            if entry != num_entries[0]:
                raise ValueError(f"Found datasets of different length at 0 ({num_entries[0]}) and {i} ({entry})")
        
        # Put all tensors in a list
        self.__data = []
        for data in datas:
            if len(data.shape) == 1:
                raise ValueError(f"The data must have at least two dimensions, since one-dimensional tensors are ambiguous. Found tensor of shape {data.shape}")
            self.__data.append(torch.as_tensor(data).float())

    # Overload print
    def __repr__(self):
        return f"Dataset{self.shape}"

    @property
    def datas(self) -> list[Tensor]:
        return self.__data
    
    # len is number of data
    def __len__(self):
        return self.num_datas
    
    @property
    def shape(self):
        return (self.num_datas,) + tuple(tuple(data.shape)[1:] for data in self.datas)
    
    @property
    def num_features(self):
        checkmul = 0
        for s in self.shape[1:]:
            a = 1
            for k in s:
                a *= k
            checkmul += a
        return checkmul
    
    @property
    def num_datas(self):
        return len(self.datas[0])
    
    # This implementation is ergonomic but need to remember lol
    # Add is concatenation of datasets
    # Sub is calculating error of datasets: absolute value of the difference
    def __add__(self, other: Dataset):
        if not isinstance(other, Dataset):
            raise TypeError(f"Does not support Dataset + {type(other)}")
        if self.num_datas != other.num_datas:
            raise NotImplementedError(f"Cannot add dataset of different number of datas ({self.num_datas} + {other.num_datas})")
        return Dataset(*self.clone().datas, *other.clone().datas)
    
    def __sub__(self, other: Dataset):
        if not isinstance(other, Dataset):
            raise TypeError(f"Does not support Dataset - {type(other)}")
        if self.shape != other.shape:
            raise NotImplementedError(f"LHS ({self.shape}) and RHS ({other.shape}) has different shape")
        diff = torch.abs(self.to_tensor() - other.to_tensor())
        return Dataset(diff).wrap(self.shape)
    
    def clone(self) -> Dataset:
        d = Dataset(*[torch.clone(data) for data in self.datas])
        return d
    
    # __iter__ automatically flattens everything and yield it as a 1D tensor
    def to_tensor(self) -> Tensor:
        return torch.cat([d.reshape(self.num_datas, -1) for d in self.datas], axis = 1)
    
    def wrap_inplace(self, shape):
        """Wraps the dataset according to the given shape. Raises an error if the numbers does not match up"""
        ## Check the length
        if shape[0] != self.num_datas:
            raise ValueError(f"The length of the shape does not match: (self: {self.num_datas}, shape: {shape[0]})")
        
        ## Do the checksum first
        check_features = 0
        for s in shape[1:]:
            a = 1
            for k in s:
                a *= k
            check_features += a

        if check_features != self.num_features:
            raise ValueError(f"The shape does not match (self: {self.num_features}, shape: {check_features})")
        
        # Flatten oneself
        orig = self.to_tensor()

        # Unwrap everything and rewrap
        i = 0
        self.__data = []
        for dims in shape[1:]:
            total = 1
            for k in dims:
                total *= k
            newshape = (shape[0],) + dims
            t = orig[:, i:i+total].view(newshape)
            i += total
            self.__data.append(t)
    
    def wrap(self, shape) -> Dataset:
        """Wraps the dataset according to the given shape. Raises an error if the numbers does not match up.
        This is a combination of the wrap_inplace method and clone method."""
        d = self.clone()
        d.wrap_inplace(shape)
        return d
    
    def split_at(self, idxs: list[int]):
        """Extracts the dataset at the specified indices."""
        shape = self.shape
        flatten = self.to_tensor()
        include_data = list(set(idxs))
        data = Dataset(flatten[include_data])
        data.wrap_inplace((len(include_data),) + shape[1:])

        exclude_data = list(set(range(len(self))) - set(idxs))
        excl_data = Dataset(flatten[exclude_data])
        excl_data.wrap_inplace((len(exclude_data),) + shape[1:])
        return data, excl_data
    
    # Gets the n data of each dataset. Does not create a clone
    def __getitem__(self, i: int | slice) -> Dataset:
        new_datas = [d[i] for d in self.datas]
        dataset = Dataset(*new_datas)
        return dataset

    # Methods to augment dataset
    def square(self):
        """Creates a new dataset with squared features."""
        poly = PolynomialFeatures(2)
        new_data = poly.fit_transform(self.to_tensor().cpu().numpy())
        new_t = torch.tensor(new_data)
        return Dataset(new_t)

    def exp(self):
        """Creates a new dataset with all the features taken exponents"""
        new_datas = [torch.exp(d) for d in self.datas]
        return Dataset(*new_datas)

class Model(ABC):
    """Base class for all models. All models have a name, a fit method, and a predict method"""
    @virtual
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        """The train logic of the model. Every model inherited from Model needs to implement this
        This needs to return the model, which will be passed in the predict_logic method as an argument"""
        raise NotImplementedError
    
    @virtual
    def predict_logic(self, model, xtest: Dataset) -> Dataset:
        """The prediction logic of the model. Every model inherited from Model needs to implement this.
        Takes in xtest and the model which will be exactly what the model has returned in fit_logic method"""
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """Returns the model name according to the class name."""
        s = self.__class__.__name__
        # Convert the class name from camel case to words
        w = []
        isup = False
        k = 0
        for i, c in enumerate(s[1:]):
            if isup and c.islower():
                isup = False
                w.append(s[k:i])
                k = i
            if c.isupper():
                isup = True
        w.append(s[k:])
        return ' '.join(w)
    
    def refresh(self):
        """Refresh the model for a new training"""
        self._trained = False
        self._informed = {}
    
    @property
    def trained(self) -> bool:
        if not hasattr(self, "_trained"):
            return False
        return self._trained
    
    @property
    def informed(self) -> dict[str, Dataset]:
        if hasattr(self, "_testing") and self._testing:
            raise ValueError("Model tried to access informed data during testing phase which is not permitted")
        if not hasattr(self, "_informed"):
            self._informed = {}
        return self._informed
    
    def inform(self, info: Dataset, name: str):
        """Informs the model about a certain information. Whether the model uses it depends on the implementation
        This informed stuff will only be available during training and not during testing"""
        self.informed[name] = info 
    
    # This is helpful for inheritance because we can only reimplement the inner logic of the fit
    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        return self.fit_logic(xtrain, ytrain)
    
    # This is helpful for inheritance becausewe can only reimplement the inner logic of the prediction
    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        return self.predict_logic(model, xtest)
    
    def fit(self, xtrain: Dataset, ytrain: Dataset):
        """Fit xtrain and ytrain on the model"""
        if xtrain.num_datas < self.min_training_data:
            raise TrainingError("Too few training data")

        if xtrain.num_features > self.max_num_features:
            raise TrainingError("Too many features")
        
        self._model = self._fit_inner(xtrain, ytrain)

        self._ytrain_shape = ytrain.shape
        self._xtrain_shape = xtrain.shape
        self._trained = True
        return self
    
    # Predict inner logic + sanity checks
    def predict(self, xtest: Dataset) -> Dataset:
        """Use xtest to predict ytest"""        
        if not self.trained:
            raise ValueError("Model has not been trained")

        if xtest.shape[1:] != self._xtrain_shape[1:]:
            a = ("_",) + self._xtrain_shape[1:]
            raise ValueError(f"Expects xtest to have shape ({a}) from model training, but got xtest with shape {xtest.shape}")
        
        # Prediction
        self._testing = True
        ypred = self._predict_inner(self._model, xtest)
        self._testing = False

        # Sanity checks
        if ypred.shape[0] != len(xtest):
            raise ValueError(f"There are different number of samples in xtest ({len(xtest)}) and ypred ({ypred.shape[0]})")
        
        # Wrap back ypred in the correct shape
        try:
            ypred.wrap_inplace((len(xtest),) + self._ytrain_shape[1:])
        except ValueError:
            raise ValueError(f"Expects ypred (shape: {ypred.shape}) and ytrain (shape: {self._ytrain_shape}) has the same number of features")
        
        return ypred
    
    def save(self, root: str):
        """Overload this if you want to save your models in the folder 'root'"""
        pass

    @property
    def max_num_features(self) -> int:
        """The max number of features of the model. This is useful when we want to constrain the number of parameters. Default is 3 million"""
        return 3000000
    
    @property
    def min_training_data(self) -> int:
        """The minimum number of training data required for the model. Default is 1"""
        return 1
    
    @property
    def model_structure(self) -> str:
        """The model structure. Default is all the non-hidden properties of the class."""
        st = ""
        for k, v in self.__dict__.items():
            if k[0] == "_":
                continue
            st += f"\n{k} = {str(v)}"
        return st.strip()
    
    @property
    def logs(self) -> str:
        """Return a unique string that identifies a particular setup of a model"""
        return f"{self.name}\n\n{self.model_structure}"

class MultiModel(Model):
    """A subclass of model where y is guaranteed to have one feature only in the implementation.
    This assumes every feature is independent and thus we can predict each model separately"""
    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        xt = xtrain.to_tensor()
        yt = ytrain.to_tensor()
        num_tasks = yt.shape[1]
        models = [None] * num_tasks
        for i in range(num_tasks):
            models[i] = self.fit_logic(xt, yt[:, i])
        return models
    
    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor()
        num_tasks = len(model)
        yp = torch.zeros(xt.shape[0], num_tasks)
        for i in range(num_tasks):
            yp[:, i] = self.predict_logic(model[i], xt)
        return yp
    
class History:
    """A helper class to plot the history of training"""
    def __init__(self):
        self.train_datas: dict[str, list] = {}
        self.test_datas: dict[str, list] = {}
        self.epochs = 0
        self.names: set[str] = set()

    def train(self, loss, name: str):
        if name not in self.names:
            self.names.add(name)
            self.train_datas[name + 'x'] = []
            self.train_datas[name + 'y'] = []
            self.test_datas[name + 'x'] = []
            self.test_datas[name + 'y'] = []
        
        self.train_datas[name + 'x'].append(self.epochs)
        self.train_datas[name + 'y'].append(loss)

    def test(self, loss, name: str):
        if name not in self.names:
            self.names.add(name)
            self.train_datas[name + 'x'] = []
            self.train_datas[name + 'y'] = []
            self.test_datas[name + 'x'] = []
            self.test_datas[name + 'y'] = []

        self.test_datas[name + 'x'].append(self.epochs)
        self.test_datas[name + 'y'].append(loss)

    def update(self):
        self.epochs += 1

    def plot(self, root: str, name: str):
        fig, ax = plt.subplots()
        for name in self.names:
            ax.plot(self.train_datas[name + 'x'], self.train_datas[name + 'y'], label = "Training " + name)
            ax.plot(self.test_datas[name + 'x'], self.test_datas[name + 'y'], label = "Test " + name)

        ax.set_yscale('log')
        ax.legend()
        ax.set_title(f"Train/Test Error plot")
        fig.savefig(f"{root}/{name} training loss.png")

# Tests
def test():
    a = np.arange(16).reshape(4, 4)
    b = np.arange(12).reshape(4, 3)

    da = Dataset(a)
    db = Dataset(b)

    dc = da + db

    assert da.shape == (4, (4,))
    assert db.shape == (4, (3,))
    assert dc.shape == (4, (4,), (3,))
    assert dc.to_tensor().shape == (4, 7)

    d = np.arange(4199).reshape((13, 17, 19))
    dd = Dataset(d)

    assert dd.shape == (13, (17, 19))
    assert dd.to_tensor().shape == (13, 17 * 19)

    try:
        da + dd
        raise AssertionError("Should not be able to add dataset of different length")
    except:
        pass

    de = Dataset(np.arange(65).reshape(13, 5))
    df = Dataset(np.arange(39).reshape(13, 3))
    dg = dd + de + df

    assert dg.shape == (13, (17, 19), (5,), (3,))
    assert dg.to_tensor().shape == (13, 17 * 19 + 5 + 3)

    # Since 17 * 19 + 5 + 3 = 331 technically we can rewrap this
    dh = dg.wrap((13, (30, 11), (1,)))
    assert dh.shape == (13, (30, 11), (1,))

    try:
        dg.wrap((13, (1, 2, 3), (4, 5, 6)))
        raise AssertionError("Should not be able to rewrap dg")
    except ValueError:
        pass

    print("Tests passed")

if __name__ == "__main__":
    test()
