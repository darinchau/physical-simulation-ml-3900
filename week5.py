import numpy as np
from models import *
from load import *
from typing import Iterable
from dataclasses import dataclass

# Root path to store all the data
ROOT = "./Datas/Week 5"

# A wrapper class for training indices
@dataclass
class TrainingIndex:
    name: str
    indices: list[int]
    
    def __iter__(self):
        return iter(self.indices)

# Trains a model
def train(model: Model, training_idxs: list[TrainingIndex]):
    vg = Dataset(np.arange(101).reshape(101, 1) / 100 * 0.0075)
    potential = Dataset(load_elec_potential())

    predictions = {}
    
    for idxs in training_idxs:
        # Split out the training data
        xtrain, ytrain, xtest, ytest = Dataset.split(vg, potential, idxs)
        
        # Train the model
        model.fit(xtrain, ytrain)

        # Test the model
        ypred = model.predict(vg)
        error = (ypred - potential).to_tensor().cpu().numpy().reshape(101, 129, 17)

        predictions[idxs.name] = np.array(error)

    return predictions

# Puts a model to the test with the given training indices
def test_model(model: Model, training_idxs: list[TrainingIndex]):
    # Train the model
    predictions = train(model, training_idxs)

    # Create logs
    path = get_folder_directory(ROOT, model)
    with open(f"{path}/logs.txt", "w", encoding="utf-8") as f:
        f.write(model.logs)

    # Save predictions
    save_h5(predictions, path)
    
if __name__ == "__main__":
    train()
