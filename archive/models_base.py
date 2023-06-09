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
from typing import Any, Iterator
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torchsummary import summary

from modules import get_device

# Filter all warnings
import warnings
warnings.filterwarnings('ignore')

__all__ = (
    "TrainingError",
    "Dataset",
    "ModelFactory",
    "Model",
    "AugmentedModel",
    "MultiModel",
    "History",
    "NeuralNetModel",
    "TimeSeriesModel",
    "TrainingIndex"
)

class TrainingError(Exception):
    """Errors raised in the Models module which are recoverable and can be simply and safely skipped over."""
    pass

# A wrapper class for training indices
@dataclass
class TrainingIndex:
    name: str
    start: int
    stop: int
    
    def __iter__(self):
        i = self.start
        while i < self.stop:
            yield i
            i += 1

    def include(self):
        return list(range(self.start, self.stop))

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
            self.__data.append(torch.as_tensor(data).double())

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
    
    def split_at(self, idxs: TrainingIndex):
        """Extracts the dataset at the specified indices. The order of data is preserved while the order of excluded_datas is not"""
        shape = self.shape
        flatten = self.to_tensor()

        include_data = idxs.include()
        data = Dataset(flatten[include_data])
        data.wrap_inplace((len(include_data),) + shape[1:])

        exclude_data = list(set(range(len(self))) - set(idxs))
        excl_data = Dataset(flatten[exclude_data])
        excl_data.wrap_inplace((len(exclude_data),) + shape[1:])
        return data, excl_data
    
    # Gets the n data of each dataset. Does not create a clone
    def __getitem__(self, i: int | slice) -> Dataset:
        if isinstance(i, int):
            if i == -1:
                i = slice(-1, None, None)
            else:
                i = slice(i, i+1, None)
        new_datas = [d[i] for d in self.datas]
        dataset = Dataset(*new_datas)
        return dataset

    # Methods to augment dataset
    def square(self):
        """Creates a new dataset with squared features."""
        # This produces all terms "up to" degree 2 - which means we have xy, x^2y, xy^2, x^2y^2
        poly = PolynomialFeatures(2)
        new_data = poly.fit_transform(self.to_tensor().cpu().numpy())
        new_t = torch.tensor(new_data)
        return Dataset(new_t)

    def nexp(self):
        """Creates a new dataset with all the features taken exponents. This gives e^-x to avoid exploding coefficients"""
        new_datas = [torch.exp(-d) for d in self.datas]
        return Dataset(*new_datas)
    
    # Append new data
    def append(self, data: Dataset) -> Dataset:
        datas = []
        for d1, d2 in zip(self.datas, data.datas):
            datas.append(torch.cat([d1, d2], axis = 0))
        return Dataset(*datas)
    
    def __iadd__(self, data: Dataset) -> Dataset:
        d = self.append(data)
        self.__data = d.datas
        return self
    
    # Unadd operation which extracts the nth data
    def extract(self, i: int) -> Dataset:
        """Extracts the ith set of data. Does not create a copy"""
        return Dataset(self.datas[i])
    
    # This allows unpacking operator
    def __iter__(self) -> Dataset:
        for d in self.datas:
            yield Dataset(d)

    # Iterates through each data in the dataset
    def into_iter(self) -> Iterator[Tensor]:
        """Iterates through the dataset"""
        for i in range(len(self)):
            yield self[i].to_tensor()

    def to(self, device):
        self.__data = [d.to(device) for d in self.datas]
        return self
    
class ModelFactory(ABC):
    """A model maker. We have a get_new method on here which returns a model. Mostly this exists as
    the OOP magic's way to make compiler checking more efficient and accurate"""
    def __init__(self):
        raise RuntimeError("Cannot initialize Model Factory on its own")
    
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
    
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self._init_args = args
        self._init_kwargs = kwargs
        return self
    
    @property
    def trained_on(self) -> list[str]:
        """A list of indices name where the model has been trained on"""
        if not hasattr(self, "_trainedon"):
            self._trainedon = []
        return self._trainedon

    # This and the __new__ method overload are required to make this python magic work
    def get_new(self, name: str) -> Model:
        """Return a fresh new instance of self with the same initialize arguments"""
        self.trained_on.append(name)
        return type(self)(*self._init_args, **self._init_kwargs)

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
        a = f"{self.name}\n\n{self.model_structure}"
        if len(self.trained_on) > 0:
            a += "\n\nTrained on:\n"
            for t in self.trained_on:
                a += f"\t{t}\n"
        return a
    
    @property
    def threads(self) -> int:
        """The maximum number of threads this should run on. Default is 4. It is just mostly a 
        suggestion but it reflects generally how much memory the training process uses
        like if training involves something like O(n^3) process then better not run this on
        too many threads"""
        return 4

# Model factory inherits from model because models can also give birth to new models
# But a model factory's only purpose is to make models
# The call hierachy for any model is:
#   fit -> _fit_inner -> base._fit_inner -> ... -> (Model)._fit_inner -> fit_logic
# This means now multiinheritance is possible
class Model(ModelFactory):
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
    
    # This prevents initialization of model factories which is not allowed
    def __init__(self):
        pass
    
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
    
    # This is helpful for inheritance because we can only reimplement the inner logic of the prediction
    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        return self.predict_logic(model, xtest)
    
    def fit(self, xtrain: Dataset, ytrain: Dataset):
        """Fit xtrain and ytrain on the model"""
        if xtrain.num_datas < self.min_training_data:
            raise TrainingError("Too few training data")

        if xtrain.num_features > self.max_num_features:
            raise TrainingError("Too many features")
        
        try:
            self._model = self._fit_inner(xtrain, ytrain)
        except ValueError as e:
            if f"{e}".startswith("Input X contains infinity"):
                raise TrainingError("Caught exploding coefficients during training")
            raise e

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
        try:
            ypred = self._predict_inner(self._model, xtest)
        except ValueError as e:
            if f"{e}".startswith("Input X contains infinity"):
                raise TrainingError("Caught exploding coefficients during training")
            raise e
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
    
    def save(self, root: str, name: str):
        """Overload this if you want to save your models in the folder 'root'"""
        pass

class AugmentedModel(Model):
    """Model but with data augmentation (square features and exponential features)"""
    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        xtrain = xtrain + xtrain.nexp()
        xtrain = xtrain + xtrain.square()
        return super()._fit_inner(xtrain, ytrain)
    
    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        xtest = xtest + xtest.nexp()
        xtest = xtest + xtest.square()
        return super()._predict_inner(model, xtest)
        

class MultiModel(Model):
    """A subclass of model where y is guaranteed to have one feature only in the implementation.
    This assumes every feature is independent and thus we can predict each model separately"""
    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        yt = ytrain.to_tensor()
        num_tasks = yt.shape[1]
        models = [None] * num_tasks
        for i in range(num_tasks):
            yt_i = Dataset(yt[:, i:i+1])
            models[i] = super()._fit_inner(xtrain, yt_i)
        return models
    
    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        num_tasks = len(model)
        yp = torch.zeros(len(xtest), num_tasks)
        for i in range(num_tasks):
            ypred_i = super()._predict_inner(model[i], xtest).to_tensor()
            yp[:, i:i+1] = ypred_i
        return Dataset(yp)
    
class History:
    """A helper class to plot the history of training"""
    def __init__(self):
        self.train_datas: dict[str, list] = {}
        self.test_datas: dict[str, list] = {}
        self.epochs = 0
        self.names: set[str] = set()
        self.logs: list[str] = []

    def train(self, loss, name: str):
        self.add_name(name)
        self.train_datas[name + 'x'].append(self.epochs)
        self.train_datas[name + 'y'].append(loss)
        self.logs.append(f"On epoch {self.epochs}, training {name} loss = {loss}")

    def test(self, loss, name: str):
        self.add_name(name)
        self.test_datas[name + 'x'].append(self.epochs)
        self.test_datas[name + 'y'].append(loss)
        self.logs.append(f"On epoch {self.epochs}, testing {name} loss = {loss}")

    def add_name(self, name):
        if name not in self.names:
            self.names.add(name)
            self.train_datas[name + 'x'] = []
            self.train_datas[name + 'y'] = []
            self.test_datas[name + 'x'] = []
            self.test_datas[name + 'y'] = []

    def update(self):
        self.epochs += 1

    def plot(self, root: str, name: str):
        fig, ax = plt.subplots()
        for n in self.names:
            ax.plot(self.train_datas[n + 'x'], self.train_datas[n + 'y'], label = "Training " + n)
            ax.plot(self.test_datas[n + 'x'], self.test_datas[n + 'y'], label = "Test " + n)

        ax.set_yscale('log')
        ax.legend()
        ax.set_title(f"Train/Test Error plot")
        fig.savefig(f"{root}/{name} training loss.jpg")

    def __iter__(self):
        for log in self.logs:
            yield log

class NeuralNetModel(Model):
    """A model except we expose the epochs argument for you, and turn on/off torch grad whenever necessary.
    NeuralNetModel must be at the base of multiinheritance hierachies"""
    @virtual
    def fit_logic(self, xtrain: Dataset, ytrain: Dataset, epochs: int = 50, verbose: bool = True) -> tuple[nn.Module, History]:
        """This should return (neural net, history object)"""
        raise NotImplementedError
    
    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset):
        self._net, self._history = self.fit_logic(xtrain, ytrain, self.epochs, self._verbose)
        return self._net

    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        with torch.no_grad():
            ypred = super()._predict_inner(model, xtest)
        return ypred
    
    def __init__(self, epochs: int, verbose: bool = True, display_every: int = 10):
        super().__init__()
        self.epochs = epochs
        self._verbose = verbose
        self._display_every = display_every
        self._device = get_device()

    def predict_logic(self, model: nn.Module, xtest: Dataset) -> Dataset:
        xt = xtest.to_tensor().to(self._device)
        output = model(xt)
        return Dataset(output)
    
    def save(self, root: str, name: str):
        self._history.plot(root, name)
        model_scripted = torch.jit.script(self._net)
        model_scripted.save(f'{root}/{name}.pt')

        with open(f"{root}/{name} history.txt", 'w') as f:
            if hasattr(self, "_logs"):
                for log in self._logs:
                    f.write(log)
                    f.write("\n")
                f.write("\n\n")
                f.write("Detailed training history:")
                for log in self._history:
                    f.write(log)
                    f.write("\n")
            f.write(summary(self._net).__repr__())

class TimeSeriesModel(Model):
    def __init__(self, use_past_n = 5):
        if use_past_n < 2:
            raise TrainingError("Use a minimum of 2 things")
        self.N = use_past_n
    
    # Let's make it only work on one-dimensional data because I don't know the best way to make it work on multidimensional data
    # To future me: This assumes datas are sorted and evenly spaced
    def _fit_inner(self, xtrain: Dataset, ytrain: Dataset) -> Any:
        N = self.N

        # Check if there is actually enough data to perform LSTM
        if len(xtrain) < N:
            raise TrainingError("Not enough data")
        
        # Check if data is one dimensional
        if xtrain.shape[1][0] != 1:
            raise ValueError("Currently, Time series model abstract class only works on one-dimensional data, which is the time")
        
        # Check if data is evenly spaced. This depends on the fact that use-last-n requires a minimum of 2
        # xt = xtrain.to_tensor().view(-1)
        # diff = xt[1] - xt[0]
        # data_diffs = torch.abs(xt[1:] - xt[:-1]) - diff
        # if not torch.all(torch.abs(data_diffs) < 1e-7):
        #     raise ValueError("Time series model has uneven spacing")
        
        fw_x = xtrain.clone()[N:]
        for i in range(1, N+1):
            fw_x = fw_x + ytrain[N-i:-i]
        fw_y = ytrain.clone()[N:]
        fw = super()._fit_inner(fw_x, fw_y)

        self._x = xtrain.clone()
        self._y = ytrain.clone()

        # Fit an extra linear component to predict everything before
        xt = xtrain.to_tensor().cpu().numpy()
        yt = ytrain.to_tensor().cpu().numpy()
        linear = Linear().fit(xt, yt)
        return fw, linear
    
    def _predict_inner(self, model, xtest: Dataset) -> Dataset:
        N = self.N
        fw, linear = model
        assert isinstance(linear, Linear)

        # Predict until cover the whole range in time steps
        diff = self._x.datas[0][1] - self._x.datas[0][0]
        xt_tensor = xtest.to_tensor()
        furthest_time_step = torch.max(xt_tensor)
        next_x = self._x.datas[0][-1][0]
        while next_x < furthest_time_step:
            next_x = next_x + diff
            new_x = Dataset(torch.tensor([[next_x]]))
            self._x += new_x
            for j in range(1, N+1):
                new_x = new_x + self._y[-j]
            ypred = super()._predict_inner(fw, new_x)
            self._y += ypred

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
            lin = Linear().fit(new_xtrain, new_ytrain)
            ypred = Dataset(lin.predict(np.array([[x]])))
            predictions.append(ypred)
        
        # Return the predictions
        if len(predictions) == 1:
            return predictions[0]
        
        d = predictions[0]
        for i in range(1, len(predictions)):
            d += predictions[i]
        return d
        

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

    dc = da + db
    dh = dc[1]
    assert dh.shape == (1, (4,), (3,))
    di = dc[1:3]
    assert di.shape == (2, (4,), (3,))

    print("Tests passed")

if __name__ == "__main__":
    test()
