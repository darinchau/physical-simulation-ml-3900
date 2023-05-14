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
from models_base import Dataset
from sklearn.linear_model import LinearRegression
from derivative import NormalizedPoissonLoss

ROOT = "./Datas/Week 7"
LATENT = 2

class Progress:
    def __init__(self, pad = 100):
        self.pad = pad
    
    def rint(self, content: str):
        print(content.ljust(self.pad), end = '\r')
        self.pad = max(self.pad, len(content) + 1)

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Sequential(
            nn.Linear(4386, 500),
            nn.Sigmoid(),
            nn.Linear(500, 50),
            nn.Sigmoid()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(50, LATENT),
            nn.Sigmoid()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(50, LATENT),
            nn.Sigmoid()
        )

        # Move device to cuda if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zero = torch.tensor(0).to(device).float()
        one = torch.tensor(1).to(device).float()
        self.N = torch.distributions.Normal(zero, one)
        self.kl = torch.tensor(0)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.linear1(x)
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(LATENT, 50),
            nn.Sigmoid(),
            nn.Linear(50, 500),
            nn.Sigmoid(),
            nn.Linear(500, 4386),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
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

def free_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"Free cuda memory: {f}")

q = 1.60217663e-19

def train(epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    ep = load_elec_potential().reshape(-1, 2193)
    sc = load_space_charge().reshape(-1, 2193)
    epsc = torch.cat([ep, sc], axis = 1).to(device).double()

    torch.cuda.empty_cache()
    net = PoissonVAE().to(device).double()
    mse = nn.MSELoss()
    poi = NormalizedPoissonLoss()
    history = []
    optimizer = torch.optim.LBFGS(net.parameters(), lr=0.01)
    p = Progress()

    for epoch in range(epochs):
        for i in range(len(epsc)):
            x = epsc[i:i+1]
            def closure():
                nonlocal history

                # Close grad
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                # fw pass
                x_hat = net(x)
                err_mse = mse(x, x_hat)
                err_kl = net.encoder.kl

                # Save loss separately 
                history.append((
                    err_mse.item(),
                    # err_poi.item(),
                    err_kl.item()
                    ))

                # bw pass
                loss = err_mse + err_kl
                if loss.requires_grad:
                    loss.backward()

                # # Since we are impatient, add this to see the logs
                nonlocal p
                mse_, kl_ = err_mse.item(), err_kl.item()
                p.rint(f"Elapsed {epoch} epochs with MSE: {mse_:.7f}, KL divergence: {kl_:.7f}")

                return loss

            optimizer.step(closure)
        
    return net, history

if __name__ == "__main__":
    net, history = train(3)
    model_scripted = torch.jit.script(net)
    model_scripted.save(f'{ROOT}/model.pt')
    hist = np.array(history)
    np.save(f"{ROOT}/history.npy", hist)
