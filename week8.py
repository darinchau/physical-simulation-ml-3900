from load import *
from torch import Tensor, nn
import torch
from model_base import Model
from modules import TrainedLinear
from derivative import NormalizedPoissonRMSE
from scipy.optimize import minimize
from anim import *
import util
import matplotlib.pyplot as plt
from load import *
from torch import Tensor, nn
import torch
from anim import *
from model_base import Model, Trainer
from modules import *
from derivative import NormalizedPoissonRMSE

ROOT = "./Datas/Week 8"

Q = 1.60217663e-19

ROOT = "./Datas/Week 8"

Q = 1.60217663e-19

class PrincipalComponentExtractor(Model):
    """Takes the tensor x, and returns the principal d dimensions by calculating its covariance matrix. d indicates the number of principal components to extract"""
    def __init__(self, d: int, /, device = None):
        self.d = d
        self.eigenvectors = None

        self._device = Device(device)

    def fit(self, X: Tensor):
        """X is a (n_data, n_features). This computes the projection data for PCA. Returns None. If you really need to access the projection data, it is at `model.pdata` and it should be a tensor with shape (N, n_features). The sorted eigenvalues is at `model.eigenvalues` and the sorted eigenvectors are at `model.eigenvectors`"""
        cov = torch.cov(X.T.float())
        l, v = torch.linalg.eigh(cov)

        self.eigenvalues, sorted_eigenidx = torch.abs(l.double()).sort(descending=True)
        self.eigenvectors = v[:, sorted_eigenidx].double()

    def project(self, X: Tensor) -> Tensor:
        """X is a (n_data, n_features) tensor. This performs the projection for you and returns an (n_data, d) tensor. Raises a runtime error if n_features does not match that in training"""
        if self.eigenvectors is None:
            raise RuntimeError("Projection data has not been calculated yet. Please first call model.fit()")
        
        if X.shape[1] != self.eigenvectors.shape[0]:
            raise RuntimeError(f"Expects {self.eigenvectors.shape[0]}-dimensional data due to training. Got {X.shape[1]}-d data instead.")
        
        components = X.double() @ self.eigenvectors[:, :self.d]
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

        X_ = torch.zeros(X.shape[0], self.eigenvalues.shape[0]).double()
        X_[:, :X.shape[1]] = X
        result = X_ @ torch.linalg.pinv(self.eigenvectors)
        return result
        
    
    def forward(self, X: Tensor) -> Tensor:
        """fit followed by project."""
        self.fit(X)
        return self.project(X)

def extract(ep, sc):
    xep = ep.reshape(-1, 129, 17)
    xsc = sc.reshape(-1, 129, 17)
            
    ep_region_2 = xep[:, 45:84,:11].reshape(-1, 429)
    ep_region_5 = xep[:, 45:84,11:].reshape(-1, 234)
    sc_region_2 = xsc[:, 45:84,:11].reshape(-1, 429)

    joined = torch.cat([ep_region_2, ep_region_5, sc_region_2], dim = 1)

    return joined

def reconstruct(x, xep, xsc):
    ep_region_2 = x[:, :429].reshape(-1, 39, 11)
    ep_region_5 = x[:, 429:663].reshape(-1, 39, 6)
    sc_region_2 = x[:, 663:].reshape(-1, 39, 11)

    xep = xep.clone()
    xep[:, 45:84,:11] = ep_region_2
    xep[:, 45:84,11:] = ep_region_5
    xep = xep.reshape(-1, 129, 17)

    xsc = xsc.clone()
    xsc[:, 45:84,:11] = sc_region_2
    xsc = xsc.reshape(-1, 129, 17)

    return xep, xsc


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
    
sc = load_space_charge() * -Q
ep = load_elec_potential()
vg = load_vgs()
ma = load_materials()
co = load_contacts()
sx, sy = load_spacing()
    
def extract(ep, sc):
    xep = ep.reshape(-1, 129, 17)
    xsc = sc.reshape(-1, 129, 17)
            
    ep_region_2 = xep[:, 45:84,:11].reshape(-1, 429)
    ep_region_5 = xep[:, 45:84,11:].reshape(-1, 234)
    sc_region_2 = xsc[:, 45:84,:11].reshape(-1, 429)

    joined = torch.cat([ep_region_2, ep_region_5, sc_region_2], dim = 1)

    return joined

def reconstruct(x, xep, xsc):
    ep_region_2 = x[:, :429].reshape(-1, 39, 11)
    ep_region_5 = x[:, 429:663].reshape(-1, 39, 6)
    sc_region_2 = x[:, 663:].reshape(-1, 39, 11)

    xep = xep.clone()
    xep[:, 45:84,:11] = ep_region_2
    xep[:, 45:84,11:] = ep_region_5
    xep = xep.reshape(-1, 129, 17)

    xsc = xsc.clone()
    xsc[:, 45:84,:11] = sc_region_2
    xsc = xsc.reshape(-1, 129, 17)

    return xep, xsc

class PoissonFixModel(Model):
    def __init__(self, device = None):
        self.fc = Sequential(
            Linear(1 + 4386 * 3, 4386),
            LeakySigmoid()
        )

    def forward(self, X: Tensor) -> Tensor:
        # This inner model should accept 1 + 4386 * 3 variables and output 4386 things
        cached_state_ = torch.zeros(3, 4386).to(Device()).double()
        results = torch.zeros(X.shape[0], 4386).to(Device()).double()

        poi = NormalizedPoissonRMSE()

        # Predict for each variable
        for i in range(X.shape[0]):
            # Predict
            x_ = torch.concat([X[i], cached_state_.reshape(-1)]).reshape(1, -1)
            x_ = self.fc(x_) # Shape should be 1, 4386

            # Gradient descent
            xep = x_[:, :2193].reshape(-1, 129, 17).detach()
            xsc = x_[:, 2193:].reshape(-1, 129, 17).detach()
            x = extract(xep, xsc).clone().detach()
            x.requires_grad = True
            optim = torch.optim.Adam([x])
            for _ in range(20):
                optim.zero_grad()
                rep, rsc = reconstruct(x, xep, xsc)
                poi_loss = poi(rep, rsc)
                poi_loss.backward()
                optim.step()

            # Reconstruct x
            rep, rsc = reconstruct(x, xep, xsc)
            x_[:, :2193] = rep.reshape(-1, 2193)
            x_[:, 2193:] = rsc.reshape(-1, 2193)
            
            # Store the cached state
            cached_state_[1:] = cached_state_[:-1].clone()
            cached_state_[0] = x_

            # Store the result
            results[i] = x_
        return results
    
from model_base import fit

class PFMTrainer(Trainer):
    def __init__(self):
        self.model = PoissonFixModel().to(Device())
        self.mse = MSELoss()
        self.poi = NormalizedPoissonRMSE()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.model(x)
        mse = self.mse(x, y)
        poi = self.poi(x[:, :2193].reshape(-1, 129, 17), x[:, 2193:].reshape(-1, 129, 17))
        self.add_loss("MSE", mse.item())
        self.add_loss("Poisson", poi.item())
        return mse + poi
    
idx = util.TRAINING_IDXS["First 30"]
model_ = PFMTrainer()
y = torch.cat([ep.reshape(-1, 2193), sc.reshape(-1, 2193)], dim = 1)
model = fit(model_, vg, y, idx, epochs=500)
