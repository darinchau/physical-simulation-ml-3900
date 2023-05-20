### Contains all different implementations of the NN modules, including custom layers, custom networks and custom activation

import gc
import torch
from torch import nn, Tensor
from load import load_spacing, get_device
from derivative import poisson_mse_, normalized_poisson_mse_
from abc import ABC, abstractmethod as virtual
from model_base import Model, ModelBase
from sklearn.linear_model import TheilSenRegressor, LinearRegression
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

class PoissonMSE(ModelBase):
    """Gives the poisson equation - the value of ||∇²φ - (-q)S||
    where S is the space charge described in p265 of the PDF 
    https://www.researchgate.net/profile/Nabil-Ashraf/post/How-to-control-the-slope-of-output-characteristicsId-Vd-of-a-GAA-nanowire-FET-which-shows-flat-saturated-region/attachment/5de3c15bcfe4a777d4f64432/AS%3A831293646458882%401575207258619/download/Synopsis_Sentaurus_user_manual.pdf"""    
    def __init__(self, device = None):
        if device is None:
            device = get_device()
        x, y = load_spacing()
        self.x = x.to(device)
        self.y = y.to(device)
    
    def forward(self, x, space_charge):
        return poisson_mse_(x, space_charge, self.x, self.y)

class NormalizedPoissonMSE(PoissonMSE):
    """Normalized means we assume space charge has already been multiplied by -q
    Gives the poisson equation - the value of sqrt(||∇²φ - (-q)S||)
    where S is the space charge described in p265 of the PDF+"""    
    def forward(self, x, space_charge):
        return normalized_poisson_mse_(x, space_charge, self.x, self.y)
    
class NormalizedPoissonRMSE(PoissonMSE):
    """Normalized means we assume space charge has already been multiplied by -q
    Gives the poisson equation - the value of sqrt(||∇²φ - (-q)S||)
    where S is the space charge described in p265 of the PDF+"""    
    def forward(self, x, space_charge):
        return torch.sqrt(normalized_poisson_mse_(x, space_charge, self.x, self.y))
    
class MSELoss(ModelBase):
    def forward(self, ypred, y) -> Tensor:
        return torch.mean((ypred - y) ** 2)
    
class ReLU(ModelBase):
    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x)
    
class Sigmoid(ModelBase):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)

class LeakySigmoid(ModelBase):
    def __init__(self, in_size):
        self.leak = nn.Linear(in_size, 1)

    def forward(self, x):
        leakage = self.leak(x)
        return leakage * x + torch.sigmoid(x)
    
# Does nothing but to make code more readable
class Identity(ModelBase):
    def forward(self, x):
        return x
    
class StochasticNode(Model):
    """Stochastic node for VAE. The output is inherently random. Specify the device if needed"""
    def __init__(self, in_size, out_size, *, device = None):
        super().__init__()

        # Use tanh for mu to constrain it around 0
        self.lmu = nn.Linear(in_size, out_size)
        self.smu = nn.Tanh()

        # Use sigmoid for sigma to constrain it to positive values and around 1
        self.lsi = nn.Linear(in_size, out_size)
        self.ssi = nn.Sigmoid()

        # Move device to cuda if possible
        if device is None:
            device = get_device()
        self.N = torch.distributions.Normal(torch.tensor(0).float().to(device), torch.tensor(1).float().to(device))
        self.kl = torch.tensor(0)

    def forward(self, x):
        mean = self.lmu(x)
        mean = self.smu(mean)

        # sigma to make sigma positive
        var = self.lsi(x)
        var = 2 * self.ssi(var)

        # z = mu + sigma * N(0, 1)
        z = mean + var * self.N.sample(mean.shape)

        # KL divergence
        # https://stats.stackexchange.com/questions/318184/kl-loss-with-a-unit-gaussian?noredirect=1&lq=1
        # https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative
        # https://kvfrans.com/variational-autoencoders-explained/
        # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048
        self.kl = -.5 * (torch.log(var) - var - mean * mean + 1).sum()

        return z

class LSTM(ModelBase):
    def __init__(self, input_dims, hidden_dims, output_dims, *, layers = 2):
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size = hidden_dims, num_layers = layers)
        self.linear = nn.Linear(hidden_dims, output_dims)

    def forward(self, x: Tensor):
        # Input is (N, features) so use strides to make it good first
        # Pretend everything is one giant sequence - one batch, 101 sequences
        out, _ = self.lstm(x)
        out = self.linear(out.view(len(x), -1))
        return out

class TrainedLinear(ModelBase):
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

# This model does not need to be trained
class PoissonJITRegressor(Model):
    """Use a first model to predict stuff, then use a second model to make them self consistent - aka satisfy the Poisson equation"""
    def __init__(self, ep1: TrainedLinear, sc1: TrainedLinear):
        # From the linearity plots, we only need to care about region 2 in practice for space charge
        # and region 2, 5 for electric potential
        self.ep1 = ep1
        self.sc1 = sc1
        
    def forward(self, x) -> Tensor:
        num_data = int(x.shape[0])
        # xep = x[:, :2193].reshape(-1, 129, 17)
        # xsc = x[:, 2193:].reshape(-1, 129, 17)

        # naive_prediction = torch.cat([self.ep1(x), self.sc1(x)], dim = 1)

        result = torch.zeros(num_data, 4386)
        with torch.no_grad():
            xep = self.ep1(x).cpu().numpy().reshape(-1, 129, 17)
            xsc = self.sc1(x).cpu().numpy().reshape(-1, 129, 17)

            poisson_loss = NormalizedPoissonRMSE('cpu')

            # Nudge region 2, 5 of ep, region 2 of sc
            # Refer to anim.py for region codes
            # The mystery numbers are the number of parameters in different region
            for i in range(num_data):
                def reconstruct(x):
                    ep_region_2 = x[:429].reshape(84 - 45, -1)
                    ep_region_5 = x[429:663].reshape(84 - 45, -1)
                    sc_region_2 = x[663:].reshape(84 - 45, -1)

                    reconstructed_ep = xep[i]
                    reconstructed_ep[45:84,:11] = ep_region_2
                    reconstructed_ep[45:84,11:] = ep_region_5
                    reconstructed_ep = torch.tensor(reconstructed_ep.reshape(1, 129, 17))

                    reconstructed_sc = xsc[i]
                    reconstructed_sc[45:84,:11] = sc_region_2
                    reconstructed_sc = torch.tensor(reconstructed_sc.reshape(1, 129, 17))

                    return reconstructed_ep, reconstructed_sc
                
                def minimize_me(x):
                    reconstructed_ep, reconstructed_sc = reconstruct(x)
                    mse = poisson_loss(reconstructed_ep, reconstructed_sc)
                    return float(mse.item())
                
                ep_region_2 = xep[i,45:84,:11].reshape(-1)
                ep_region_5 = xep[i,45:84,11:].reshape(-1)
                sc_region_2 = xsc[i,45:84,:11].reshape(-1)

                joined = np.concatenate([ep_region_2, ep_region_5, sc_region_2])
                bounds = [(0, 1)] * 663 + [(-20, 20)] * 429
                gradient_descent = minimize(minimize_me, x0 = joined, bounds = bounds)
                grad_result = gradient_descent.x
                new_ep, new_sc = reconstruct(grad_result)
                result[i][:2193] = new_ep.reshape(-1)
                result[i][2193:] = new_sc.reshape(-1)

                # print(f"Frame {i}: Difference: {torch.mean(torch.abs(naive_prediction[i] - result[i]))}", end = "")

                # poi = poisson_loss(new_ep, new_sc)
                # print(f" Poisson loss: {poi}")

        return result
