import numpy as np
from models import *
from load import *
from dataclasses import dataclass
import multiprocessing as mp
from anim import make_plots
from multiprocessing import Pool
from trainer import Trainer

# Root path to store all the data
ROOT = "./Datas/Week 7"

if __name__ == "__main__":
    Trainer(ROOT).test_all_models([
        LinearModel()
    ])