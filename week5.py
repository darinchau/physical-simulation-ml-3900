import numpy as np
from models import *
from load import *

def train():
    vg = Dataset(np.arange(101) / 100 * 0.0075)
    potential = Dataset(load_elec_potential())
    xtrain, ytrain, xtest, ytest = Dataset.split(vg, potential, range(20))
    
    model = LinearModel()
    model.fit(xtrain, ytrain)
    ypred = model.predict(vg)

    errors = (vg - potential).to_tensor().cpu().numpy().reshape(101, 129, 17)
