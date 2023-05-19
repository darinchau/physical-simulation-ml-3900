# A high-level pytorch thing I hacked together in an hour

from __future__ import annotations
import torch
from torch import nn, Tensor
import pickle
from typing import Optional, Self
from load import get_device
import matplotlib.pyplot as plt
from tqdm import trange
import time
from abc import ABC, abstractmethod as virtual

class Model(ABC):
    """Abstract class for all models/model layers etc"""
    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls)
        # Calls super init here so that no one forgets
        super().__init__(self)

        self._init_args = args
        self._init_kwargs = kwargs

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
        raise NotImplementedError
    
    def __init__(self):
        """Creates a new model that can be combined with other models to form bigger models. :)"""
        pass
    
    # A recursive save
    def _serialize(self) -> dict:
        state = {}
        for objname, obj in self._model_children():
            state[objname] = obj._serialize()
        state["_=freezed"] = self._freezed
        return state
    
    def _deserialize(self, state: dict):
        for k, v in state.items():
            if "_=" in k:
                continue
            self.__getattr__(k)._deserialize(v)
        if state["_=freezed"]:
            self.freeze()
        return self
    
    def _model_children(self):
        """Return a generator of (object name, all children model). This is not recursive"""
        for objname in dir(self):
            try:
                obj = self.__getattr__(objname)
            except AttributeError:
                continue
            if isinstance(obj, Model):
                yield objname, obj

    # Defines a way to save the model
    def save(self, path: str):
        """Save the model file. We suggest to use the '.hlpt' extension for high-level pytorch"""
        state = self._serialize()
        state["_init_=args"] = self._init_args
        state["_init_=kwargs"] = self._init_kwargs
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self = cls(*state["_init_=args"], **state["_init_=kwargs"])
        self._deserialize(state)
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

    def __repr__(self):
        return self._class_name()

    def freeze(self):
        """Freezes the model so it is no longer trainable"""
        self._freezed = True
        for _, p in self._model_children():
            p.freeze()
        return self

    def __call__(self, *x):
        if self._freezed:
            self.eval()
        
        return self.forward(*x)
    
class Trainer(Model):
    """A special type of model designed to train other models. This provides the `self.history` attribute in which one can log the losses"""
    @property
    def history(self):
        if not hasattr(self, '_history'):
            self._history = History()
        return self._history
    
    @property
    def training(self):
        if not hasattr(self, '_training'):
            self._training = False
        return self._training
    
    def update(self):
        self.history.update()

    def train(self):
        self._training = True

    def test(self):
        self._training = False

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
        xtrain: Tensor, 
        ytrain: Tensor, 
        xtest: Tensor, 
        ytest: Tensor, 
        optim: torch.optim.Optimizer = None, 
        device: torch.device | str = None,
        epochs: int = 100
    ) -> Trainer:
    """Provides us with the sklearn like interface which abstracts away the training loop. However, a dedicated trained has to be created. 
    Default optimizer is Adam with lr = 0.005
    Default device is whatever is available on your PC
    Default epochs is 100
    Batch size is not available because we dont need it
    
    Returns the trained trainer"""
    if optim is None:
        optim = torch.optim.Adam(model.parameters(), lr = 0.005)

    if device is None:
        device = get_device()

    net = model.to(device).double()
    xtrain = xtrain.to(device).double()
    xtest = xtest.to(device).double()
    ytrain = ytrain.to(device).double()
    ytest = ytest.to(device).double()

    # Show a lovely progress bar

    epochs_pbar = trange(100)

    for epoch in epochs_pbar:
        net.train()
        trainloss = 0.
        for x, y in zip(xtrain, ytrain):
            def closure():
                if torch.is_grad_enabled():
                    optim.zero_grad()

                loss = net(x, y)

                if loss.requires_grad:
                    loss.backward()

                nonlocal trainloss
                trainloss += loss.item()
                
                return loss
            optim.step(closure)
        
        net.test()
        testloss = 0.
        with torch.no_grad():
            for (x, y) in zip(xtest, ytest):
                loss = net(x, y)
                testloss += loss.item()

        log = f"On epoch {epoch}, train loss = {float(trainloss/len(xtrain)):.7f}, test loss = {float(testloss/len(xtest)):.7f}"
        epochs_pbar.set_description(log)

    return net

class ModelBase(Model):
    """Models base objects are layers directly from pytorch. One must define the model that it is trying to inherit. Serialize gives state dict and there are no module children for us to loop over"""
    @property
    @virtual
    def model(self) -> nn.Module:
        """Returns (a reference to) the model"""
        raise NotImplementedError
    
    def _serialize(self) -> dict:
        return {
            "state_dict": self.model.state_dict(),
            "_=freezed": self._freezed
        }
    
    def _deserialize(self, state: dict):
        self.model.load_state_dict(state["state_dict"])
        if state["_=freezed"]:
            self.freeze()

    def freeze(self):
        self._freezed = True
        for x in self.model.parameters():
            x.requires_grad = False

    def _num_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _num_nontrainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

    def _get_model_info(self, layers: int):
        s = "- " * layers + f"{self._class_name()} (Trainable: {self._num_trainable()}, Other: {self._num_nontrainable()})"
        return s
    
    def _model_children(self):
        raise NotImplementedError("Model children is not recursive in nature. One must implement all base cases if one inherits from model base")
    
    def forward(self, x: Tensor) -> Tensor:
        m = self.model
        x = m(x)
        return x

##### Tests #####
import os
import random

class LinearTestModel(ModelBase):
    def __init__(self, ins, outs):
        self.fc = nn.Linear(ins, outs)

    @property
    def model(self) -> nn.Module:
        return self.fc
    
class LinearTestModel2(Model):
    def __init__(self, linear):
        self.fc = linear

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        return x

class LinearTestModel3(Model):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self, *x: Tensor) -> Tensor:
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
    os.remove(path)

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
    os.remove(path)

    print("Test 4 passed")

    a4.fc.freeze()
    a4.save(path)
    a5 = LinearTestModel2.load(path)
    assert a5.fc._freezed
    os.remove(path)

    print("Test 5 passed")

    a6 = LinearTestModel3(a, a3)
    c = a6(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    a6.save(path)
    a7 = LinearTestModel3.load(path)
    c = a7(t)
    assert c.shape == (178, 200)
    assert torch.all(b == c)

    print("Test 6 passed")

if __name__ == "__main__":
    test()
