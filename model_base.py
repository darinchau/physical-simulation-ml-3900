# A high-level pytorch thing I hacked together in an hour

from __future__ import annotations
import torch
from torch import nn, Tensor
import pickle

# Overload nn module for a high-level api for myself
class Model(nn.Module):
    """Base class for all models/model layers etc"""
    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls)
        # Calls super init here so that no one forgets
        super().__init__(self)
        self._init_args = args
        self._init_kwargs = kwargs
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
        for objname, obj in self._model_children():
            state[objname] = obj._serialize()
        state["_init_=args"] = self._init_args
        state["_init_=kwargs"] = self._init_kwargs
        return state
    
    def _deserialize(self, state: dict):
        for k, v in state.items():
            if k in ("_init_=args", "_init_=kwargs"):
                continue
            self.__getattr__(k)._deserialize(v)
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
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> Model:
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
    
    def __repr__(self):
        line =  "=" * 100
        return f"""{self._class_name()}\nModel summary:
{line}
{self._get_model_info(0)}
{line}
"""

class ModelBase(Model):
    """Models base objects are layers directly inherited from pytorch. Serialize gives state dict and there are no module children for us to loop over"""
    def _serialize(self) -> dict:
        return self.state_dict()
    
    def _deserialize(self, state: dict):
        self.load_state_dict(state)

    def _num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _num_nontrainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def _get_model_info(self, layers: int):
        s = "- " * layers + f"{self._class_name()} (Trainable: {self._num_trainable()}, Other: {self._num_nontrainable()})"
        return s
    
    def _model_children(self):
        raise StopIteration