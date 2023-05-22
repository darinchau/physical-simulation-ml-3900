### Contains all different implementations of the NN modules, including custom layers, custom networks and custom activation

from __future__ import annotations
import torch
from torch import nn, Tensor
from load import get_device
from abc import ABC, abstractmethod as virtual
from model_base import Model, ModelBase
from util import straight_line_model
import numpy as np
from scipy.optimize import minimize

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
        self.fc = nn.Linear(in_size, out_size)
    
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
        self.inited = False

    def forward(self, x):
        if not self.inited:
            self.leak = nn.Linear(x.shape[1], 1)
            self.inited = True
        leakage = self.leak(x)
        return leakage * x + torch.sigmoid(x)
    
    def _num_trainable(self) -> int:
        if self.inited:
            return sum(p.numel() for p in self.leak.parameters() if p.requires_grad)
        return 0
    
    def eval(self):
        if self.inited:
            self.leak.eval()
    
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
        if device is None:
            device = get_device()

        # Use tanh for mu to constrain it around 0
        self.lmu = Linear(in_size, out_size)
        self.smu = Tanh()

        # Use sigmoid for sigma to constrain it to positive values and around 1
        self.lsi = Linear(in_size, out_size)
        self.ssi = Sigmoid()

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

class LSTM(Model):
    """An lstm node."""
    _info_show_impl_details = False

    def __init__(self, input_dims, hidden_dims, output_dims, *, layers = 2):
        self.out_layers = output_dims
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size = hidden_dims, num_layers = layers)
        self.linear = nn.Linear(hidden_dims, output_dims)

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

class Cached(Model):
    """Caches the previous output and uses it for the next prediction. The only difference with an LSTM is we save the prediction state instead of a separate hidden state
    This has the advantage of forcing the model to remember only the last N results.
    
    If the original should take in `in` features and output `out` features, then
    - `fc`: A model that takes in `in` + `N` * `out`
    - `N` is the number of results we want to cache"""
    _info_show_impl_details = False
    def __init__(self, fc: Model, N: int, / , device = None):
        self.N = N
        if device is None:
            self._device = get_device()
        else:
            self._device = device
        self.inited = False
        self.fc = fc
    
    def forward(self, x: Tensor) -> Tensor:
        # Input shape here is always (n terms in sequence, n_features)
        cached_state_ = torch.zeros((self.N, self.out_size)).to(self._device).double()
        
        results = torch.zeros((x.shape[0], self.out_size))

        for i in range(x.shape[0]):
            # Predict and then cache the output state
            x_ = torch.concat([x[i], results.reshape(-1)])
            y = self.fc(x_)
            cached_state_[1:] = cached_state_[:-1]
            cached_state_[0] = y
            results[i] = y

        return results

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
    
    def _class_name(self) -> str:
        return f"Cached (N = {self.N})"

