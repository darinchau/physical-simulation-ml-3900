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
import warnings

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

class PoissonMSE(Model):
    """Gives the poisson equation - the value of ||∇²φ - (-q)S||
    where S is the space charge described in p265 of the PDF 
    https://www.researchgate.net/profile/Nabil-Ashraf/post/How-to-control-the-slope-of-output-characteristicsId-Vd-of-a-GAA-nanowire-FET-which-shows-flat-saturated-region/attachment/5de3c15bcfe4a777d4f64432/AS%3A831293646458882%401575207258619/download/Synopsis_Sentaurus_user_manual.pdf"""    
    def __init__(self, device = None):
        if device is None:
            self._device = get_device()
        else:
            self._device = device
        x, y = load_spacing()
        self._x = x.to(device)
        self._y = y.to(device)
    
    def forward(self, x, space_charge):
        return poisson_mse_(x, space_charge, self._x, self._y)

class NormalizedPoissonMSE(PoissonMSE):
    """Normalized means we assume space charge has already been multiplied by -q
    Gives the poisson equation - the value of sqrt(||∇²φ - (-q)S||)
    where S is the space charge described in p265 of the PDF"""    
    def forward(self, x, space_charge):
        return normalized_poisson_mse_(x, space_charge, self._x, self._y)
    
class NormalizedPoissonRMSE(PoissonMSE):
    """Normalized means we assume space charge has already been multiplied by -q
    Gives the poisson equation - the value of sqrt(||∇²φ - (-q)S||)
    where S is the space charge described in p265 of the PDF"""    
    def forward(self, x, space_charge):
        return torch.sqrt(normalized_poisson_mse_(x, space_charge, self._x, self._y))
    
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

class PrincipalComponentExtractor(Model):
    """Takes the tensor x, and returns the principal d dimensions by calculating its covariance matrix. d indicates the number of principal components to extract"""
    def __init__(self, d: int, /, device = None):
        self.d = d
        self.eigenvectors = None

        if device is None:
            self._device = get_device()
        else:
            self._device = device

    def fit(self, X: Tensor):
        """X is a (n_data, n_features). This computes the projection data for PCA. Returns None. If you really need to access the projection data, it is at `model.pdata` and it should be a tensor with shape (N, n_features). The sorted eigenvalues is at `model.eigenvalues` and the sorted eigenvectors are at `model.eigenvectors`"""
        cov = torch.cov(X.T.float())
        l, v = torch.linalg.eig(cov)

        # Temporarily supress warnings. This normally screams at us for discarding the complex part. But X.T @ X is always positive definite so real eigenvalues :)
        warnings.filterwarnings("ignore", category=UserWarning) 
        self.eigenvalues, sorted_eigenidx = torch.abs(l.double()).sort(descending=True)
        self.eigenvectors = v[:, sorted_eigenidx].double()
        warnings.filterwarnings("default", category=UserWarning)

    def project(self, X: Tensor) -> Tensor:
        """X is a (n_data, n_features) tensor. This performs the projection for you and returns an (n_data, d) tensor. Raises a runtime error if n_features does not match that in training"""
        if self.eigenvectors is None:
            raise RuntimeError("Projection data has not been calculated yet. Please first call model.fit()")
        
        P = self.eigenvectors[:, :self.d]
        
        if X.shape[1] != P.shape[0]:
            raise RuntimeError(f"Expects {P.shape[0]}-dimensional data due to training. Got {X.shape[1]}-d data instead.")
        
        # Welcome to transpose hell
        X_ = X - torch.mean(X.float(), dim = 0)
        components = X_.double() @ P
        return components
    
    def unproject(self, X: Tensor):
        """Try to compute the inverse of model.project(X). The input is a tensor of shape (n_data, d) and returns a tensor of (n_data, n_features)"""
        # XP = X* so given X* we have X = X*P⁻¹
        # Problem is P is a matrix of shape (n_features, d), so we need to make it square first to take inverse.
        # However, P is originally (n_features, n_features) big which we can take inverses, the reason
        # P has the shape (n, d) is because it is actually the combination of the real P matrix followed by extracting first N columns
        # We use a workaround: append zeros on X until it has enough features, then use the full P inverse
        if self.eigenvectors is None:
            raise RuntimeError("Projection data has not been calculated yet. Please first call model.fit()")

        X_ = torch.zeros(X.shape[0], self.eigenvalues.shape[0])
        X_[:, :X.shape[1]] = X
        X_ = X_.double()
        P = self.eigenvectors
        try:
            result = X_ @ torch.linalg.inv(P)
        except RuntimeError as e:
            # raise RuntimeError(f"PCA eigenvectors matrix is not invertible for some reason. This is probably due to that there are very very very small (coerced to 0) eigenvalues.\n\nOriginal error: \"\"\"\n{e}\n\"\"\"")
            # Use a psuedoinverse if the inverse is not available
            # Also A @ B = (B.T @ A.T).T
            result =  torch.linalg.lstsq(P.T, X_.T).solution.T
        return result
        
    
    def forward(self, X: Tensor) -> Tensor:
        """fit followed by project."""
        self.fit(X)
        return self.project(X)
