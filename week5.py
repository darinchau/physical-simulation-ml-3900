import numpy as np
from models import *
from load import *
from typing import Iterable

def train(model: Model, to_test: list[Iterable[int]]):
    vg = Dataset(np.arange(101).reshape(101, 1) / 100 * 0.0075)
    potential = Dataset(load_elec_potential())
    xtrain, ytrain, xtest, ytest = Dataset.split(vg, potential, range(20))
    
    # Train the model
    model = LinearModel()
    model.fit(xtrain, ytrain)

    # Test the model
    ypred = model.predict(vg)
    errors = (ypred - potential).to_tensor().cpu().numpy().reshape(101, 129, 17)
    

if __name__ == "__main__":
    train()
