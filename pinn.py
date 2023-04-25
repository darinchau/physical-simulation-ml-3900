## This module contains code for physics-informed neural networks, motivated by the first 4 weeks of stuff in archive/models.py

from __future__ import annotations
from sklearn.linear_model import LinearRegression as Linear
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from abc import abstractmethod as virtual
from load import index_exclude, index_include
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary
from numpy import log
import matplotlib.pyplot as plt
import re
import math
from scipy.optimize import minimize
from numba import njit
