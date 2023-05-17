### Contains all different implementations of the NN modules, including custom layers, custom networks and custom activation

import gc
import torch
from torch import nn, Tensor
from load import load_spacing, get_device
from derivative import poisson_mse_, normalized_poisson_mse_
from abc import ABC, abstractmethod as virtual

class PoissonNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 100),
            nn.Sigmoid(),
            nn.Linear(100, 2193),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class PoissonLoss(nn.Module):
    """Gives the poisson equation - the value of ||∇²φ - (-q)S||
    where S is the space charge described in p265 of the PDF 
    https://www.researchgate.net/profile/Nabil-Ashraf/post/How-to-control-the-slope-of-output-characteristicsId-Vd-of-a-GAA-nanowire-FET-which-shows-flat-saturated-region/attachment/5de3c15bcfe4a777d4f64432/AS%3A831293646458882%401575207258619/download/Synopsis_Sentaurus_user_manual.pdf"""    
    def __init__(self):
        super().__init__()
        self.device = get_device()
        x, y = load_spacing()
        self.x = x.to(self.device)
        self.y = y.to(self.device)
    
    def forward(self, x, space_charge):
        return poisson_mse_(x, space_charge, self.x, self.y)

class NormalizedPoissonMSE(PoissonLoss):
    """Normalized means we assume space charge has already been multiplied by -q
    Gives the poisson equation - the value of sqrt(||∇²φ - (-q)S||)
    where S is the space charge described in p265 of the PDF+"""    
    def forward(self, x, space_charge):
        return normalized_poisson_mse_(x, space_charge, self.x, self.y)

# A leaky sigmoid function
class NotSigmoid(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.leak = nn.Linear(in_size, 1)

    def forward(self, x):
        leakage = self.leak(x)
        return leakage * x + torch.sigmoid(x)
    
# Does nothing but to make code more readable
class Identity(nn.Module):
    def forward(self, x):
        return x
    
class StochasticNode(nn.Module):
    def __init__(self, input, output):
        super().__init__()

        # Use tanh for mu to constrain it around 0
        self.lmu = nn.Linear(input, output)
        self.smu = nn.Tanh()

        # Use sigmoid for sigma to constrain it to positive values and around 1
        self.lsi = nn.Linear(input, output)
        self.ssi = nn.Sigmoid()

        # Move device to cuda if possible
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

# Idk why I cant get an LSTM layer to work except like this
class LSTMLayer(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, *, layers = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_dims, num_layers = layers)
        self.linear = nn.Linear(hidden_dims, output_dims)

    def forward(self, x: Tensor):
        # Input is (N, features) so use strides to make it good first
        # Pretend everything is one giant sequence - one batch, 101 sequences
        out, _ = self.lstm(x)
        out = self.linear(out.view(len(x), -1))
        return out
    
class VAEModule(nn.Module, ABC):
    """Interface for VAE Modules"""
    @virtual
    def forward(self, x):
        raise NotImplementedError

    @virtual
    def get_kl_divergence(self):
        raise NotImplementedError
    
    @virtual
    def encode(self, x):
        raise NotImplementedError
    
    @virtual
    def decode(self, x):
        raise NotImplementedError

    def get_latent_dims(self) -> int:
        raise NotImplementedError


class PoissonVAE(VAEModule):
    def __init__(self, latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4386, 512),
            Identity(),
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh()
        )

        self.latent = latent

        self.stochastic_node = StochasticNode(16, latent)

        self.decoder = nn.Sequential(
            nn.Linear(latent, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh()
        )

        self.ep_decode = nn.Sequential(
            nn.Linear(64, 256),
            nn.Sigmoid(),
            nn.Linear(256, 2193),
            nn.Sigmoid()
        )

        self.sc_decode = nn.Sequential(
            nn.Linear(64, 256),
            NotSigmoid(256),
            nn.Linear(256, 2193),
            NotSigmoid(2193)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.stochastic_node(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        ep = self.ep_decode(x)
        sc = self.sc_decode(x)
        x = torch.cat([ep, sc], dim = -1)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def get_kl_divergence(self):
        return self.stochastic_node.kl
    
    def get_latent_dims(self) -> int:
        return self.latent
    
class PoissonVAE2(VAEModule):
    def __init__(self, latent) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            LSTMLayer(4386, 1000, 512),
            NotSigmoid(512),
            LSTMLayer(512, 128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh()
        )

        self.latent = latent

        self.stochastic_node = StochasticNode(16, latent)

        self.decoder = nn.Sequential(
            nn.Linear(latent, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh()
        )

        self.ep_decode = nn.Sequential(
            nn.Linear(64, 256),
            nn.Sigmoid(),
            nn.Linear(256, 2193),
            nn.Sigmoid()
        )

        self.sc_decode = nn.Sequential(
            nn.Linear(64, 256),
            NotSigmoid(256),
            nn.Linear(256, 2193),
            NotSigmoid(2193)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.stochastic_node(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        ep = self.ep_decode(x)
        sc = self.sc_decode(x)
        x = torch.cat([ep, sc], dim = -1)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def get_kl_divergence(self):
        return self.stochastic_node.kl
    
    def get_latent_dims(self) -> int:
        return self.latent
    
class SymmetricNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 100),
            nn.Sigmoid(),
            nn.Linear(100, 17 * 72),
            nn.Sigmoid()
        )
        self.x_spacing = load_spacing()[0]
        self.device = get_device()

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.reshape(-1, 72, 17)
        total = 0.078
        
        target = torch.zeros(x.shape[0], 129, 17).to(self.device).double()
        target[:, :72, :] = x

        for j in range(72, 129):
            x_pos = total - self.x_spacing[j]

            # Use normal flip near the edges
            if total - x_pos < 0.02:
                col = torch.abs(x_pos - self.x_spacing).argmin()
                target[:, j, :] = x[:, col, :]
                continue
            
            # Use lerp near the center
            col1 = torch.searchsorted(self.x_spacing, x_pos, side='right') - 1
            col2 = col1 + 1
            weighting = (self.x_spacing[col2] - x_pos)/(self.x_spacing[col2] - self.x_spacing[col1])
            target[:, j, :] = weighting * x[:, col1, :] + (1-weighting) * x[:, col2, :]
        
        return target.reshape(-1, 2193)

class VAENet1(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.net: VAEModule = mod
        for p in self.net.parameters():
            p.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(1, 3),
            nn.Linear(3, 5)
        )

    def forward(self, x):
        self.net.eval()
        x = self.fc(x)
        x = self.net.decode(x)
        return x
