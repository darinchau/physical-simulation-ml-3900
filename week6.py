import numpy as np
from models import *
from load import *
from dataclasses import dataclass
import multiprocessing as mp
from anim import make_plots
from multiprocessing import Pool
from trainer import Trainer

# Root path to store all the data
ROOT = "./Datas/Week 6"

if __name__ == "__main__":
    # Trainer(ROOT).test_all_models([
    #     LinearAugmentedModel(),
    #     LinearAugmentedLSTMModel(),
    #     LinearLSTMModel(),
    #     RidgeAugmentedModel(),
    # ])

    # Trainer(ROOT).test_all_models([
    #     BayesianRegressionModel(),
    #     StochasticLSTMModel(use_past_n=5),
    # ])

    Trainer(ROOT).test_all_models([
        PoissonModel(epochs=200, display_every=1),
        SymmetricPoissonModel(epochs=200, display_every=1)
    ])
