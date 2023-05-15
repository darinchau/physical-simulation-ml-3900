import sys
import gc
import numpy as np
from models import *
from load import *
from dataclasses import dataclass
import multiprocessing as mp
from anim import make_plots, AnimationMaker, DataVisualizer
from multiprocessing import Pool
from trainer import Trainer
from torch import Tensor, nn
import torch
import matplotlib.pyplot as plt
from models_base import Dataset, get_device
from sklearn.linear_model import LinearRegression
from derivative import NormalizedPoissonRMSE

ROOT = "./Datas/Week 7"

class Progress:
    def __init__(self, pad = 100):
        self.pad = pad
    
    def rint(self, content: str):
        print(content.ljust(self.pad), end = '\r')
        self.pad = max(self.pad, len(content) + 1)

LATENT = 2

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(4386, 300)
        self.s1 = nn.Sigmoid()
        self.l2 = nn.Linear(300, 50)
        self.s2 = nn.Sigmoid()

        self.lmu = nn.Linear(50, LATENT)
        self.smu = nn.Sigmoid()

        self.lsi = nn.Linear(50, LATENT)
        self.ssi = nn.Sigmoid()

        # Move device to cuda if possible
        device = get_device()
        zero = torch.tensor(0).float().to(device)
        one = torch.tensor(1).float().to(device)
        self.N = torch.distributions.Normal(zero, one)
        self.kl = torch.tensor(0)

    def forward(self, x):
        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Linear 1 + normalization + activation
        x = self.l1(x)
        x = self.s1(x)

        # Linear 2 + normalization + activation
        x = self.l2(x)
        x = self.s2(x)

        # mu + normalization
        mu = self.lmu(x)
        mu = self.smu(mu)

        # sigma + normalization + exp to make sigma positive
        sigma = self.lsi(x)
        sigma = self.ssi(sigma)
        sigma = torch.exp(sigma)

        # z = mu + sigma * N(0, 1)
        z = mu + sigma * self.N.sample(mu.shape)

        # KL divergence
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(LATENT, 50)
        self.s1 = nn.Sigmoid()
        self.l2 = nn.Linear(50, 300)
        self.s2 = nn.Sigmoid()
        self.l3 = nn.Linear(300, 4386)
        self.s3 = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.s1(x)

        x = self.l2(x)
        x = self.s2(x)

        x = self.l3(x)
        x = self.s3(x)
        return x

class PoissonVAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

Q = 1.60217663e-19

# Use this with the debugger to create an ad hoc cuda memory watcher in profile txt
class CudaMonitor:
    # Property flag forces things to save everytime a line of code gets run in the debugger
    @property
    def memory(self):
        s = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    # Total numbers
                    total = 1
                    for i in obj.shape:
                        total *= i
                    s.append((total, f"Tensor: {type(obj)}, size: {obj.size()}, shape: {obj.shape}"))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                pass
        s = [x[1] for x in sorted(s, key = lambda a: a[0], reverse = True)]
        with open("profile.txt", 'w') as f:
            f.write(f"Memory allocated: {torch.cuda.memory_allocated()}\n")
            f.write(f"Max memory allocated: {torch.cuda.max_memory_allocated()}\n")
            for y in s:
                f.write(y)
                f.write("\n")
        return "\n".join(s)
    
    def clear(self):
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    del obj
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                pass

def load():
    ep = Dataset(load_elec_potential())
    sc = Dataset(load_space_charge() * (-Q))
    epsc = (ep + sc).clone().to_tensor().reshape(-1, 1, 4386)
    print(epsc.shape)
    return epsc

def train(epochs: int):
    device = get_device()
    net = PoissonVAE().to(device).double()
    history = []
    mse = nn.MSELoss()
    poi = NormalizedPoissonRMSE()
    epsc = load().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    p = Progress()
    c = CudaMonitor()

    for epoch in range(epochs):
        for x in epsc:
            def closure():
                # Bring memory watcher in :)
                nonlocal c

                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                x_hat = net(x)
                mse_loss = mse(x, x_hat)
                poi_loss = poi(x_hat[:, :2193], x_hat[:, 2193:])
                kl_diver = net.encoder.kl
                loss = mse_loss + poi_loss + kl_diver
                if loss.requires_grad:
                    loss.backward()

                nonlocal history
                history.append([mse_loss.item(), poi_loss.item(), kl_diver.item()])

                nonlocal p
                mse_, poi_, kl_ = history[-1]
                p.rint(f"Elapsed {epoch} epochs with MSE: {mse_:.7f}, Poisson: {poi_:.7f}, KL divergence: {kl_:.7f}")

                return loss
            optimizer.step(closure)

if __name__ == "__main__":
    train(10)
