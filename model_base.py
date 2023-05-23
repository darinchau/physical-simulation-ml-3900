# A high-level pytorch thing I hacked together in an hour

from __future__ import annotations
import torch
from torch import nn, Tensor
import pickle
from typing import Any, Optional, Self
import matplotlib.pyplot as plt
from tqdm import trange
import time
from abc import ABC, abstractmethod as virtual
from load import Device
import ast
from util import TrainingIndex

def get(self, attr_name):
    try:
        return self.__dict__[attr_name]
    except KeyError:
        try:
            return self.__getattr__(attr_name)
        except AttributeError:
            return None

def _call_fix(f):
    # This fixes the docs on __call__
    for doc in (
        f.__doc__,
        f.__class__.forward.__doc__,
        f.__class__.__doc__,
        Model.forward.__doc__
    ):
        if doc is None:
            continue
        f.__doc__ = doc
        return f
    raise NotImplementedError

class Model(ABC):
    """Abstract class for all models/model layers etc"""
    # If true, then recursively show the model children details in summary
    _info_show_impl_details = True

    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls)
        # Calls super init here so that no one forgets
        super().__init__(self)

        # Save the init args and kwargs
        # self._init_args = args
        # self._init_kwargs = kwargs
        self._extra_members = {}

        # One of the points of doing this is to not call super().__init__() every time when we define our own module
        # So interrupt stuff here
        self._freezed = False

        try:
            self.forward()
        except NotImplementedError:
            raise AttributeError("Forward method not found")
        except Exception as e:
            # Not the place to raise exceptions here
            pass
        
        return self
    
    @virtual
    def forward(self, *x: Tensor) -> Tensor:
        """Passes the tensor forward"""
        raise NotImplementedError
    
    def __init__(self):
        """Creates a new model that can be combined with other models to form bigger models. :)"""
        pass
    
    def _model_children(self):
        """Return a generator of (object name, all children model). This is not recursive"""
        _warned = False
        for objname in dir(self):
            obj = get(self, objname)

            if obj is None:
                continue

            if isinstance(obj, nn.Module) and not _warned:
                print(f"Found an nn.Module inside a model (type: {self._class_name()}). While this is alright, saving and loading this model cannot retreive the state of this module. Use a ModelBase to wrap this nn.Module instead.")
                _warned = True

            if isinstance(obj, Model):
                yield objname, obj

    def __setattr__(self, __name: str, __value: Any) -> None:
        # avoid recursion
        if __name != "_extra_members" or __name.endswith("_"):
            self._extra_members[__name] = __value
        return super().__setattr__(__name, __value)

    def _pickel(self):
        state = {}
        for objname, obj in self._model_children():
            state[objname] = (obj.__class__, obj._pickel())

        # Serialize all other members
        state["_=members"] = self._extra_members
        return state
    
    def _unpickel(self, state: dict[str, tuple[type, dict]]):
        for k, v in state["_=members"].items():
            self.__setattr__(k, v)
        
        for k, v in state.items():
            if k.count("_="):
                continue
            cls, child_state = v[0], v[1]
            child_obj = cls.__new__(cls)
            child_obj._unpickel(child_state)
            self.__setattr__(k, child_obj)
        return self

    # Defines a way to save the model
    def save(self, path: str):
        """Save the model file. We suggest to use the `.hlpt` extension for high-level pytorch"""
        state = self._pickel()
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> Self:
        """Loads the model"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self = cls.__new__(cls)
        self._unpickel(state)
        if not isinstance(self, cls):
            raise TypeError("Unpickled model has a different type than the type specified.")
        return self
    
    def _class_name(self) -> str:
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
    
    def _num_trainable(self) -> int:
        total = 0
        for _, model in self._model_children():
            total += model._num_trainable()
        return total
    
    def _num_nontrainable(self) -> int:
        total = 0
        for _, model in self._model_children():
            total += model._num_nontrainable()
        return total
    
    # Recursively get all the model info
    def _get_model_info(self, layers: int):
        s = "- " * layers + f"{self._class_name()} (Trainable: {self._num_trainable()}, Other: {self._num_nontrainable()})"
        if self._info_show_impl_details:
            for _, model in self._model_children():
                s += "\n"
                s += model._get_model_info(layers + 1)
        return s
    
    def summary(self):
        line =  "=" * 100
        return f"""{self._class_name()}\nModel summary:
{line}
{self._get_model_info(0)}
{line}
"""

    def to(self, device):
        for _, p in self._model_children():
            p.to(device)
        return self

    def double(self):
        return self.to(torch.float64)

    def float(self):
        return self.to(torch.float32)

    def __repr__(self):
        return self._class_name()

    def freeze(self):
        """Freezes the model so it is no longer trainable"""
        self._freezed = True
        for _, p in self._model_children():
            p.freeze()
        return self
    
    def eval(self):
        """Put the model in evaluation mode"""
        for _, p in self._model_children():
            p.eval()

    def __call__(self, *x):
        if self._freezed:
            self.eval()
        
        return self.forward(*x)
    
    def _parameters(self):
        for _, p in self._model_children():
            for param in p._parameters():
                yield param
    
class ModelBase(Model):
    """Models base objects are layers directly from pytorch. Serialize gives state dict and there are no module children for us to loop over"""
    _info_show_impl_details = False
    
    @property
    def model(self) -> nn.Module:
        """Returns (a reference to) the model. Defaults to looping over the directory and finding the one and only one model. If there is more than one model, this raises a TypeError"""
        # Try to find nn modules as data member
        if not hasattr(self, "_model"):
            for objname in dir(self):
                obj = get(self, objname)

                if obj is None:
                    continue

                # We do not allow models as members because this causes issues in saving
                if isinstance(obj, Model):
                    raise TypeError(f"Object (type: {self._class_name()}) must not have Models as data members")
                
                # If we found a module, set it to _model. If something has already been set, raise an error
                if isinstance(obj, nn.Module):
                    if hasattr(self, "_model"):
                        raise TypeError(f"There are more than one nn module defined here at (type: {self._class_name()})")
                    self._model = obj, objname
            
            # If we cannot find a single nn module as data member
            if not hasattr(self, "_model"):
                raise TypeError(f"Model (type: {self._class_name()}) base has no nn module as data member")
        
        if not isinstance(self._model[0], nn.Module):
            raise TypeError(f"self._model must be a pytorch module. Found type: {type(self._model[0]).__name__}")
        
        return self._model[0]
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def _pickel(self):
        # First get name to the model reference
        # implicitly call the model method to create the self._model attribute
        _ = self.model
        model, name = self._model
        state = {
            "model": model,
            "name": name,
            "state_dict": self.model.state_dict(),
            "_=freezed": self._freezed
        }
        return state
    
    def _unpickel(self, state):
        model, name = state["model"], state["name"]
        self.__setattr__(name, model)
        self.model.load_state_dict(state["state_dict"])
        if state["_=freezed"]:
            self.freeze()

    def freeze(self):
        self._freezed = True
        for x in self.model.parameters():
            x.requires_grad = False

    def eval(self):
        """Puts the model in evaluation mode"""
        self.model.eval()

    def _num_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _num_nontrainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
    
    def _model_children(self):
        raise NotImplementedError(f"Model (type: {self._class_name()}) children is not recursive in nature. One must implement all base cases if one inherits from model base")
    
    def _parameters(self):
        for p in self.model.parameters():
            yield p

class Trainer(Model):
    """A special type of model designed to train other models. This provides the `self.history` attribute in which one can log the losses
    
    `add_loss()`, `log()` avaiable"""
    @property
    def history(self) -> History:
        if not hasattr(self, '_history'):
            self._history = History()
        return self._history
    
    @property
    def training(self):
        if not hasattr(self, '_training'):
            self._training = False
        return self._training
    
    def _update(self):
        self.history.update()

    def _train(self):
        self._training = True

    def _test(self):
        self._training = False

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Takes in x and y and returns the loss. Keep in mind a `self.history` History object is available"""
        raise NotImplementedError

    def __call__(self, x, y) -> Tensor:        
        er = super().__call__(x, y)

        try:
            if er.shape != ():
                raise RuntimeError(f"Trainer must return a single pytorch scalar. Found tensor of shape {er.shape} instead")
        except AttributeError:
            raise RuntimeError(f"Trainer must return a single pytorch scalar. Found {type(er).__name__} instead")
        
        return er
    
    def add_loss(self, name: str, loss: float):
        if self.training:
            name = 'Train ' + name
        else:
            name = 'Test ' + name

        self.history.add_loss(name, loss)

    def log(self, s: str):
        self.history.logs.append(s)
    
class History:
    """A helper class to plot the history of training"""
    def __init__(self):
        self.losses: list[dict[str, float]] = [{}]
        self.counts = {}
        self.names: set[str] = set()
        self.logs: list[str] = []

    def add_loss(self, name, loss):
        self.names.add(name)
        
        if name in self.counts:
            n = self.counts[name]
            self.losses[-1][name] = (n * self.losses[-1][name] + loss) / (n + 1)
            self.counts[name] += 1
        else:
            self.losses[-1][name] = loss
            self.counts[name] = 1

    def update(self):
        self.losses.append({})
        self.counts = {}

    def plot(self, root: str, name: str):
        fig, ax = plt.subplots()
        for name in self.names:
            x, y = [], []
            for i, losses in enumerate(self.losses):
                if name in losses:
                    x.append(i)
                    y.append(losses[name])
            ax.plot(x, y, label = name)

        ax.set_yscale('log')
        ax.legend()
        ax.set_title(f"Train/Test Error plot")
        fig.savefig(f"{root}/{name} training loss.jpg")

    def __iter__(self):
        for i, losses in enumerate(self.losses):
            s = f"On epoch {i}: "
            s += ", ".join([f"{k} = {v:.6f}"for k, v in losses.items()])
            yield s
        
        yield "\n"
        for s in self.logs:
            yield s

def fit(
        model: Trainer, 
        x: Tensor, 
        y: Tensor,
        idx: TrainingIndex,
        optim: torch.optim.Optimizer = None, 
        epochs: int = 100
    ) -> Trainer:
    """Provides us with the sklearn like interface which abstracts away the training loop. However, a dedicated trained has to be created. 
    Default optimizer is Adam with lr = 0.005
    Default device is whatever is available on your PC
    Default epochs is 100
    Batch size is not available because we dont need it
    
    Returns the trained trainer"""
    if optim is None:
        optim = torch.optim.Adam(model._parameters(), lr = 0.005)

    device = Device()
    net = model.to(device).double()
    xtrain = x[idx].to(device).double()
    xtest = x.to(device).double()
    ytrain = y[idx].to(device).double()
    ytest = y.to(device).double()

    # Show a lovely progress bar

    epochs_pbar = trange(epochs)

    for epoch in epochs_pbar:
        net._train()
        trainloss = 0.
        for x, y in zip(xtrain, ytrain):
            def closure():
                if torch.is_grad_enabled():
                    optim.zero_grad()

                loss = net(x.reshape(1, -1), y.reshape(1, -1))

                if loss.requires_grad:
                    loss.backward()

                nonlocal trainloss
                trainloss += loss.item()
                
                return loss
            optim.step(closure)
        
        net._test()
        testloss = 0.
        for (x, y) in zip(xtest, ytest):
            loss = net(x.reshape(1, -1), y.reshape(1, -1))
            testloss += loss.item()

        log = f"On epoch {epoch + 1}, train loss = {float(trainloss/len(xtrain)):.7f}, test loss = {float(testloss/len(xtest)):.7f}"
        epochs_pbar.set_description(log)

    return net

##### Tests #####
import os
import random

class LinearTestModel(ModelBase):
    def __init__(self, ins, outs):
        self.fc = nn.Linear(ins, outs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        return x
    
class LinearTestModel2(Model):
    def __init__(self, linear):
        self.fc = linear

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        return x

class LinearTestModel3(Model):
    def __init__(self, a, b, c = 5):
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        x = self.a(x)
        return x
    
def test():
    a = LinearTestModel(100, 200)
    t = torch.randn(178, 100)
    b = a(t)
    assert b.shape == (178, 200)

    j = random.randint(0, 10000)
    path = f"./temp_model_test{j}"

    print("Test 1 passed")

    a.save(path)
    a2 = LinearTestModel.load(path)
    c = a2(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    print("Test 2 passed")
    
    a3 = LinearTestModel2(a)
    c = a3(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    print("Test 3 passed")

    a3.save(path)
    a4 = LinearTestModel2.load(path)
    c = a4(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    print("Test 4 passed")

    a4.fc.freeze()
    a4.save(path)
    a5 = LinearTestModel2.load(path)
    assert a5.fc._freezed

    print("Test 5 passed")

    a6 = LinearTestModel3(a, a3, c = 6)
    c = a6(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    a6.save(path)
    a7 = LinearTestModel3.load(path)
    c = a7(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    assert a7.c == 6

    print("Test 6 passed")

    class LinearTestModel4(Model):
        def __init__(self, linear):
            self.fc = linear

        def forward(self, x: Tensor) -> Tensor:
            x = self.fc(x)
            return x

    a8 = LinearTestModel4(a)
    c = a8(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    a8.save(path)
    a9 = LinearTestModel4.load(path)
    c = a9(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    print("Test 7 passed")

    os.remove(path)

if __name__ == "__main__":
    test()
