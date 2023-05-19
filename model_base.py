# A high-level pytorch thing I hacked together in an hour

from __future__ import annotations
import torch
from torch import nn, Tensor
import pickle
from typing import Optional, Self

from torch.nn.modules.module import Module

# Overload nn module for a high-level api for myself
class Model(nn.Module):
    """Base class for all models/model layers etc"""

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
        """Return a generator of all children model"""
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
        for p in self.parameters():
            p.requires_grad = False
        return self

    def __call__(self, *x):
        if self._freezed:
            self.eval()
        
        return super().__call__(*x)

class ModelBase(Model):
    """Models base objects are layers directly inherited from pytorch. Serialize gives state dict and there are no module children for us to loop over"""
    def _serialize(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "_=freezed": self._freezed
        }
    
    def _deserialize(self, state: dict):
        self.load_state_dict(state["state_dict"])
        if state["_=freezed"]:
            self.freeze()

    def _num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _num_nontrainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def _get_model_info(self, layers: int):
        s = "- " * layers + f"{self._class_name()} (Trainable: {self._num_trainable()}, Other: {self._num_nontrainable()})"
        return s
    
    def _model_children(self):
        raise StopIteration


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

if __name__ == "__main__":
    test()
