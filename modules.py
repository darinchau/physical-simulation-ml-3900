### Contains all different implementations of the NN modules, including custom layers, custom networks and custom activation

from __future__ import annotations
import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod as virtual
from model_base import Model, ModelBase
from util import straight_line_model
import numpy as np
from scipy.optimize import minimize
from load import Device

class Sequential(Model):
    """Sequential model"""
    def __init__(self, *f):
        for model in f:
            assert isinstance(model, Model)
        self.fs = f

    def forward(self, x):
        for model in self.fs:
            x = model(x)
        return x
    
    def _model_children(self):
        for i, model in enumerate(self.fs):
            yield str(i), model
    
    def _deserialize(self, state: dict):
        for k, v in state.items():
            if k in ("_init_=args", "_init_=kwargs"):
                continue
            self.fs[int(k)]._deserialize(v)
        return self

class Linear(ModelBase):
    """Linear layer"""
    def __init__(self, in_size: int, out_size: int):
        self.fc = nn.Linear(in_size, out_size).double()
        torch.nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, x):
        x = self.fc(x)
        return x
    
class MSELoss(Model):
    def forward(self, ypred, y) -> Tensor:
        return torch.mean((ypred - y) ** 2)
    
class ReLU(Model):
    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x)
    
class Sigmoid(Model):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)

class LeakySigmoid(Model):
    _info_show_impl_details = False

    def __init__(self):
        self.inited = None
        self.tos = []

    def forward(self, x):
        if self.inited is None:
            self.leak = Linear(x.shape[1], 1)
            self.inited = x.shape[1]
            for device in self.tos:
                self.leak.to(device)
        elif x.shape[1] != self.inited:
            raise RuntimeError(f"X shape is not consistent. Expects {self.inited} features from last pass but got {x.shape} features")
        leakage = self.leak(x)
        return leakage * x + torch.sigmoid(x)
    
    def to(self, device):
        if not self.inited:
            self.tos.append(device)
        else:
            self.leak.to(device)
            
class Tanh(Model):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x)
    
# Does nothing but to make code more readable
class Identity(Model):
    def forward(self, x):
        return x
    
class StochasticNode(Model):
    """Stochastic node for VAE. The output is inherently random. Specify the device if needed"""
    _info_show_impl_details = False

    def __init__(self, in_size, out_size, *, device = None):
        # Use tanh for mu to constrain it around 0
        self.lmu = Linear(in_size, out_size)
        self.smu = Tanh()

        # Use sigmoid for sigma to constrain it to positive values and around 1
        self.lsi = Linear(in_size, out_size)
        self.ssi = Sigmoid()

        device = Device(device)

        # Move device to cuda if possible
        self._N = torch.distributions.Normal(torch.tensor(0).float().to(device), torch.tensor(1).float().to(device))
        self._kl = torch.tensor(0)

        self.eval_ = False

    def forward(self, x):
        mean = self.lmu(x)
        mean = self.smu(mean)

        # sigma to make sigma positive
        var = self.lsi(x)
        var = 2 * self.ssi(var)

        # In evaluation mode, return the mean directly
        if self.eval_:
            return mean

        # z = mu + sigma * N(0, 1)
        z = mean + var * self._N.sample(mean.shape)

        # KL divergence
        # https://stats.stackexchange.com/questions/318184/kl-loss-with-a-unit-gaussian?noredirect=1&lq=1
        # https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative
        # https://kvfrans.com/variational-autoencoders-explained/
        # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048
        self._kl = -.5 * (torch.log(var) - var - mean * mean + 1).sum()

        return z
    
    def eval(self):
        self.eval_ = True
        self.lmu.eval()
        self.smu.eval()

class LSTM_(ModelBase):
    def __init__(self, i, h, l):
        self.lstm = nn.LSTM(input_size=i, hidden_size=h, num_layers=l)
        torch.nn.init.xavier_uniform_(self.lstm.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.lstm(x)

class LSTM(Model):
    """An lstm node."""
    _info_show_impl_details = False

    def __init__(self, input_dims, hidden_dims, output_dims, *, layers = 2):
        self.out_layers = output_dims
        self.lstm = LSTM_(input_dims, hidden_dims, layers)
        self.linear = Linear(hidden_dims, output_dims)

    def forward(self, x: Tensor):
        out, _ = self.lstm(x)
        out = self.linear(out.view(len(x), -1))
        return out
    
    def __call__(self, x):
        """Input is 
            (N, n_features) - we will pretend the whole thing is one giant sequence, or 
            (N, n terms in sequence, n_features): the sequences are separated :)
            
        Output is:
            (N, n_features_out) or (N, n terms in sequence, n_out_features) respectively"""
        if len(x.shape) == 3:
            results = torch.zeros(x.shape[0], x.shape[1], self.out_layers)
            for i in range(x.shape[0]):
                results[i] = super().__call__(x[i])
            return results
        return super().__call__(x)
    
    def _get_model_info(self, layers: int):
        s = "- " * layers + f"{self._class_name()} (Trainable: {self._num_trainable()}, Other: {self._num_nontrainable()})"
        return s
    
    def _num_trainable(self):
        return sum(p.numel() for p in self.lstm.parameters() if p.requires_grad) + sum(p.numel() for p in self.linear.parameters() if p.requires_grad)
    
    def _num_nontrainable(self):
        return sum(p.numel() for p in self.lstm.parameters() if not p.requires_grad) + sum(p.numel() for p in self.linear.parameters() if not p.requires_grad)
    
    def eval(self):
        self.lstm.eval()
        self.linear.eval()

class TrainedLinear(Model):
    """A linear node except the weights and biases are pretrained separately"""
    _info_show_impl_details = False
    def __init__(self, in_size, out_size, algorithm = 'TheilSen'):
        self._trained = False
        self.coefs = nn.Parameter(torch.zeros(in_size, out_size), requires_grad=False)
        self.intercept = nn.Parameter(torch.zeros(out_size), requires_grad=False)

        # algorithms
        self.model = straight_line_model(algorithm)
    
    def fit(self, x: Tensor, y: Tensor):
        num_features = x.shape[1]
        num_tasks = y.shape[1]
        if num_features != self.coefs.shape[0]:
            raise ValueError(f"Incorrect in_size. Expected {self.coefs.shape[0]} but got {num_features}")
        
        if num_tasks != self.coefs.shape[1]:
            raise ValueError(f"Incorrect out_size. Expected {self.coefs.shape[1]} but got {num_tasks}")

        with torch.no_grad():
            for i in range(num_tasks):
                xtrain = x.cpu().numpy()
                ytrain = y[:, i].cpu().numpy()
                model = self.model.fit(xtrain, ytrain)
                self.coefs[:, i] = torch.tensor(model.coef_)
                self.intercept[i] = torch.tensor(model.intercept_)
        
        self._trained = True
        self.freeze()
        return self
    
    def forward(self, x):
        if not self._trained:
            raise ValueError("Trained linear layer has not been trained.")
        x = x @ self.coefs + self.intercept
        return x
    
    def _num_nontrainable(self) -> int:
        return self.intercept.numel() + self.coefs.numel()
    
    def eval(self):
        pass
    
class LSTMStack(Model):
    """Reshapes the model before feeding into an LSTM or cached linear (etc.) model. The inner implementation is just a reshape inside a try-catch block"""
    def __init__(self, stack_size: int):
        self._stack_size = stack_size

    def forward(self, x: Tensor) -> Tensor:
        try:
            return x.reshape(-1, self._stack_size, x.shape[1])
        except RuntimeError as e:
            if e.args[0].startswith("shape"):
                raise ValueError(f"The resize stack size {self._stack_size} is invalid for {x.shape[0]} datas")
    
class Flatten(Model):
    """Reshapes the tensor back to (N, n_features)"""
    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(1, -1)
