import numpy as np
from models import *
from load import *
from typing import Iterable
from dataclasses import dataclass
import multiprocessing as mp

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
        ypred = model.predict(vg).to_tensor().cpu().numpy().reshape(101, 129, 17)

        # Save the prediction
        predictions[idxs.name] = np.array(ypred)

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
    save_h5(predictions, f"{path}/predictions.h5")

# Test each model. If there is only one, then use sequential, otherwise parallel
def test_all_models(models: list[Model], training_idxs: list[TrainingIndex]):
    if len(models) == 1:
        test_model(models[0], training_idxs)
        return
    
    with mp.Pool(processes = 4) as pool:
        pool.starmap(test_model, [(model, training_idxs) for model in models])
    
if __name__ == "__main__":
    test_all_models([
        LinearModel(),
        GaussianModel(),
    ], training_idxs = [
        TrainingIndex("First 5", range(5)),
        TrainingIndex("First 20", range(20)),
        TrainingIndex("First 30", range(30)),
        TrainingIndex("First 40", range(40)),
        TrainingIndex("First 60", range(60)),
        TrainingIndex("First 75", range(75)),
        TrainingIndex("First 90", range(90)),
        TrainingIndex("15 to 45", range(15, 45)),
        TrainingIndex("20 to 40", range(20, 40)),
        TrainingIndex("40 to 60", range(40, 60)),
        TrainingIndex("25 to 35", range(25, 35)),
        TrainingIndex("20 to 50", range(20, 50)),
        TrainingIndex("30 to 50", range(30, 50)),
        TrainingIndex("29 and 30 and 31", [29, 30, 31])
    ])
