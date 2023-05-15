### Contains all different implementations of the NN modules, including custom layers, custom networks and custom activation

import torch
from torch import nn, Tensor
from load import load_spacing
from derivative import poisson_mse_, normalized_poisson_mse_

# Get Cuda if cuda is available
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # return torch.device('cpu')

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

class NormalizedPoissonRMSE(PoissonLoss):
    """Normalized means we assume space charge has already been multiplied by -q
    Gives the poisson equation - the value of sqrt(||∇²φ - (-q)S||)
    where S is the space charge described in p265 of the PDF 
    https://www.researchgate.net/profile/Nabil-Ashraf/post/How-to-control-the-slope-of-output-characteristicsId-Vd-of-a-GAA-nanowire-FET-which-shows-flat-saturated-region/attachment/5de3c15bcfe4a777d4f64432/AS%3A831293646458882%401575207258619/download/Synopsis_Sentaurus_user_manual.pdf"""    
    def forward(self, x, space_charge):
        return normalized_poisson_mse_(x, space_charge, self.x, self.y)

# A leaky sigmoid function
class NotSigmoid(nn.Module):
    def __init__(self, leakage = .1):
        super().__init__()
        device = get_device()
        self.leakage = leakage * torch.nn.Parameter(torch.ones(1)).to(device)

    def forward(self, x):
        return self.leakage * x + torch.sigmoid(x)
    
# Does nothing but to make code more readable
class Identity(nn.Module):
    def forward(self, x):
        return x

# Idk why I cant get an LSTM layer to work except like this
class LSTMLayer(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_dims, batch_first=True)
        self.linear = nn.Linear(hidden_dims, output_dims)
        
    def forward(self, x):
        out, hidden = self.lstm(x.unsqueeze(0))
        out = self.linear(hidden[0].squeeze(0))
        return out

class PoissonVAEEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.l1 = LSTMLayer(4386, 1000, 1000)
        self.s1 = Identity()
        self.l2 = nn.Linear(1000, 300)
        self.s2 = NotSigmoid()
        self.l3 = nn.Linear(300, 50)
        self.s3 = nn.Sigmoid()

        # Use tanh for mu to constrain it around 0
        self.lmu = nn.Linear(50, latent_dims)
        self.smu = nn.Tanh()

        # Use sigmoid for sigma to constrain it to positive values and around 1
        self.lsi = nn.Linear(50, latent_dims)
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

        # Linear layers
        x, _ = self.l1(x)
        x = self.s1(x)

        x = self.l2(x)
        x = self.s2(x)

        x = self.l3(x)
        x = self.s3(x)

        # mu + normalization
        mean = self.lmu(x)
        mean = self.smu(mean)

        # sigma to make sigma positive
        var = self.lsi(x)
        var = self.ssi(var)

        # z = mu + sigma * N(0, 1)
        z = mean + var * self.N.sample(mean.shape)

        # KL divergence
        # https://stats.stackexchange.com/questions/318184/kl-loss-with-a-unit-gaussian?noredirect=1&lq=1
        # https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative
        # https://kvfrans.com/variational-autoencoders-explained/
        # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048
        self.kl = -.5 * (torch.log(var) - var - mean * mean + 1).sum()
        return z

class PoissonVAEDecoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super().__init__()
        self.l1 = nn.Linear(latent_dims, 50)
        self.s1 = nn.Sigmoid()
        self.l2 = nn.Linear(50, 300)
        self.s2 = nn.Sigmoid()

        self.l3sc = nn.Linear(300, 1000)
        self.s3sc = NotSigmoid(0.5)
        self.l4sc = nn.Linear(1000, 2193)
        self.s4sc = NotSigmoid(0.5)

        self.l3ep = nn.Linear(300, 500)
        self.s3ep = nn.Sigmoid()
        self.l4ep = nn.Linear(500, 2193)
        self.s4ep = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.s1(x)

        x = self.l2(x)
        x = self.s2(x)

        ep = self.l3ep(x)
        ep = self.s3ep(ep)

        ep = self.l4ep(ep)
        ep = self.s4ep(ep)
        
        sc = self.l3sc(x)
        sc = self.s3sc(sc)

        sc = self.l4sc(sc)
        sc = self.s4sc(sc)

        x = torch.cat([ep, sc], dim = -1)
        return x

class PoissonVAE(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
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