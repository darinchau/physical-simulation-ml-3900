import sys
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
    # Who wrote this garbage
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(4386, 300)
        self.s1 = nn.Tanh()
        self.l2 = nn.Linear(300, 50)
        self.s2 = nn.Tanh()

        self.lmu = nn.Linear(50, LATENT)
        self.smu = nn.Tanh()

        self.lsi = nn.Linear(50, LATENT)
        self.ssi = nn.Tanh()

        # Move device to cuda if possible
        device = get_device()
        zero = torch.tensor(0).float().to(device)
        one = torch.tensor(1).float().to(device)
        self.N = torch.distributions.Normal(zero, one)
        self.kl = torch.tensor(0)

    def forward(self, x):
        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Linear 1 + normalization + tanh activation
        x = self.l1(x)
        mx = torch.max(torch.abs(x))
        x = self.s1(x/mx)*mx

        # Linear 2 + normalization + tanh
        x = self.l2(x)
        mx = torch.max(torch.abs(x))
        x = self.s2(x/mx)*mx

        # mu + normalization
        mu = self.lmu(x)
        mx = torch.max(torch.abs(mu))
        mu = self.smu(mu/mx)*mx

        # sigma + normalization + exp to make sigma positive
        sigma = self.lsi(x)
        mx = torch.max(torch.abs(sigma))
        sigma = self.ssi(sigma/mx)*mx
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
        self.s1 = nn.Tanh()
        self.l2 = nn.Linear(50, 300)
        self.s2 = nn.Tanh()
        self.l3 = nn.Linear(300, 4386)
        self.s3 = nn.Tanh()

    def forward(self, x):
        x = self.l1(x)
        mx = torch.max(torch.abs(x))
        x = self.s1(x/mx)*mx

        x = self.l2(x)
        mx = torch.max(torch.abs(x))
        x = self.s2(x/mx)*mx

        x = self.l3(x)
        mx = torch.max(torch.abs(x))
        x = self.s3(x/mx)*mx
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

def train(epochs: int):
    ep = Dataset(load_elec_potential())
    sc = Dataset(load_space_charge() * (-Q))
    epsc = (ep + sc).clone().to_tensor().reshape(-1, 1, 4386)
    print(epsc.shape)

    device = get_device()
    net = PoissonVAE().to(device).double()
    history = []
    mse = nn.MSELoss()
    poi = NormalizedPoissonRMSE()
    epsc = epsc.to(device)
    optimizer = torch.optim.LBFGS(net.parameters(), lr=0.01)
    p = Progress()

    for epoch in range(epochs):
        for x in epsc:
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                x_hat = net(x)
                mse_loss = mse(x, x_hat)
                poi_loss = poi(x[:, :2193], x[:, 2193:])
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
