## This module contains code for base classes of models, motivated by the first 4 weeks of stuff in archive/models.py
## But most implementations in the train.py file is a bit too complicated now, so we start over with a different code structure

from __future__ import annotations
from sklearn.linear_model import LinearRegression as Linear
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod as virtual, ABC
import torch
from torch import Tensor
from typing import Any

# Filter all warnings
import warnings

from models_base import Dataset
warnings.filterwarnings('ignore')

__all__ = (
    "TrainingError",
    "Dataset",
    "Model"
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
    
    @staticmethod
    def split(x: Dataset, y: Dataset, train_index: list[int]):
        """Performs train test split"""
        if len(x) != len(y):
            raise ValueError(f"Cannot split data because x ({len(x)}) and y ({len(y)}) has different length")

        xshape = x.shape
        yshape = y.shape

        # This creates copies already so don't worry
        xflatten = x.to_tensor()
        yflatten = y.to_tensor()

        include_data = list(set(train_index))
        xtrain = Dataset(xflatten[include_data])
        xtrain.wrap_inplace((len(include_data),) + xshape[1:])

        ytrain = Dataset(yflatten[include_data])
        ytrain.wrap_inplace((len(include_data),) + yshape[1:])

        exclude_data = list(set(range(len(x))) - set(train_index))
        xtest = Dataset(xflatten[exclude_data])
        xtest.wrap_inplace((len(exclude_data),) + xshape[1:])

        ytest = Dataset(yflatten[exclude_data])
        ytest.wrap_inplace((len(exclude_data),) + yshape[1:])

        return xtrain, ytrain, xtest, ytest
    
    # Gets the n data of each dataset. Does not create a clone
    def __getitem__(self, i: int | slice) -> Dataset:
        new_datas = [d[i] for d in self.datas]
        dataset = Dataset(*new_datas)
        return dataset
    
    def to_device(self):
        # Move everything to cuda if necessary
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__data = [d.to(device) for d in self.datas]

class Model(ABC):
    """Base class for all models. All models have a name, a fit method, and a predict method"""
    @virtual
    def fit_logic(self, xtrain: Tensor, ytrain: Tensor) -> Any:
        """The train logic of the model. Every model inherited from Model needs to implement this
        This needs to return the model, which will be passed in the predict_logic method as an argument"""
        raise NotImplementedError
    
    @virtual
    def predict_logic(self, model, xtest: Tensor) -> Tensor:
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
    
    # Initialize the model
    def __init__(self):
        self._trained = False
    
    @property
    def trained(self) -> bool:
        if not hasattr(self, "_trained"):
            return False
        return self._trained
    
    # This is helpful for inheritance because we can only reimplement the inner logic of the fit
    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        xt = xtrain.to_tensor()
        yt = ytrain.to_tensor()
        return self.fit_logic(xt, yt)
    
    # This is helpful for inheritance becausewe can only reimplement the inner logic of the prediction
    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor()
        return Dataset(self.predict_logic(model, xt))
    
    def fit(self, xtrain: Dataset, ytrain: Dataset):
        """Fit xtrain and ytrain on the model"""
        self.__init__()

        if xtrain.num_datas < self.min_training_data:
            raise TrainingError("Too few training data")

        if xtrain.num_features > self.max_num_features:
            raise TrainingError("Too many features")
        
        self._model = self._fit_inner(xtrain, ytrain)

        self._ytrain_shape = ytrain.shape
        self._xtrain_shape = xtrain.shape
        self._trained = True
    
    # Predict inner logic + sanity checks
    def predict(self, xtest: Dataset) -> Dataset:
        """Use xtest to predict ytest"""
        if not self.trained:
            raise ValueError("Model has not been trained")

        if xtest.shape[1:] != self._xtrain_shape[1:]:
            a = ("_",) + self._xtrain_shape[1:]
            raise ValueError(f"Expects xtest to have shape ({a}) from model training, but got xtest with shape {xtest.shape}")
        
        # Prediction
        ypred = self._predict_inner(self._model, xtest)

        # Sanity checks
        if ypred.shape[0] != len(xtest):
            raise ValueError(f"There are different number of samples in xtest ({len(xtest)}) and ypred ({ypred.shape[0]})")
        
        # Wrap back ypred in the correct shape
        try:
            ypred.wrap_inplace((len(xtest),) + self._ytrain_shape[1:])
        except ValueError:
            raise ValueError(f"Expects ypred (shape: {ypred.shape}) and ytrain (shape: {self._ytrain_shape}) has the same number of features")
        
        return ypred

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
    
class RecurrentModel(Model):
    def __init__(self, use_last_n: int = 5):
        super().__init__()
        self.use_last_n = use_last_n

    @property
    def min_training_data(self) -> int:
        return self.use_last_n + 1

    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        # Adjoin the last n ytrain to xtrain
        N = self.use_last_n
        new_x = xtrain[N:]
        for i in range(1, N+1):
            new_x = new_x + ytrain[N-i:-i]
        xt = new_x.to_tensor()
        yt = ytrain.to_tensor()
        model = self.fit_logic(xt, yt)
        return model
    
    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        pass
        

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
