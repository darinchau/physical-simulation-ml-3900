# A high-level pytorch thing I hacked together in an hour

from __future__ import annotations
import torch
from torch import nn, Tensor
import pickle

# Overload nn module for a high-level api for myself
class Model(nn.Module):
    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls)
        # Calls super init here so that no one forgets
        super().__init__(self)
        try:
            self.forward()
        except NotImplementedError:
            raise AttributeError("Forward method not found")
        except Exception as e:
            pass
        return self
    
    def forward(self, *x: Tensor) -> Tensor:
        raise NotImplementedError
    
    # A recursive save
    def _serialize(self) -> dict:
        state = {}
        for objname in dir(self):
            obj = self.__getattr__(objname)
            if isinstance(obj, Model):
                state[objname] = obj._serialize()
        return state
    
    def _deserialize(self, state: dict):
        for k, v in state.items():
            self.__getattr__(k)._deserialize(v)
        return self

    # Defines a way to save the model
    def save(self, path: str):
        """Save the model file. We suggest to use the '.hlpt' extension for high-level pytorch"""
        state = self._serialize()
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self._deserialize(state)
        return self
    
class ModelBase(Model):
    """Models base objects are layers directly inherited from pytorch. The only difference is serialize should give the state dict"""
    def _serialize(self) -> dict:
        return self.state_dict()
    
    def _deserialize(self, state: dict):
        self.load_state_dict(state)
