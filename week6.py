import numpy as np
from models import *
from load import *
from dataclasses import dataclass
import multiprocessing as mp
from anim import make_plots
from multiprocessing import Pool

# Root path to store all the data
ROOT = "./Datas/Week 6"

# A wrapper class for training indices
@dataclass
class TrainingIndex:
    name: str
    indices: list[int]
    
    def __iter__(self):
        return iter(self.indices)
    
def train(model: Model, idx: TrainingIndex, root: str):
    # Get the datas
    print(f"Starting training for {model.name} on {idx.name}")
    vg = Dataset(np.arange(101).reshape(101, 1) / 100 * 0.75)
    potential = Dataset(load_elec_potential())
    edensity = Dataset(load_e_density())
    space_charge = Dataset(load_space_charge())

    # Refresh the model
    model = model.get_new(idx.name)

    # Split out the training data
    xtrain, xtest = vg.split_at(idx)
    ytrain, ytest = potential.split_at(idx)
    edtrain, edtest = edensity.split_at(idx)
    sctrain, sctest = space_charge.split_at(idx)
    
    # Perform informed training
    model.inform(edtrain, "edensity")
    model.inform(edtest, "edensity-test")
    model.inform(sctrain, "spacecharge")
    model.inform(sctest, "spacecharge-test")
    model.inform(xtest, "xtest")
    model.inform(ytest, "ytest")

    # Train the model
    try:
        model.fit(xtrain, ytrain)
    except TrainingError as e:
        return None
    
    # Test the model
    ypred = model.predict(vg).to_tensor().cpu().numpy().reshape(101, 129, 17)

    # Save the model
    model.save(root, idx.name)

    # Print logs
    print(f"Finished training {model.name} with {idx.name}")

    return idx.name, np.array(ypred)

# Puts a model to the test with the given training indices
def test_model(model: Model, training_idxs: list[TrainingIndex]):
    # Create folder
    path = get_folder_directory(ROOT, model)
    predictions = {}

    # Train model for each index
    with Pool(processes=4) as pool:
        results = pool.starmap(train, [(model, idx, path) for idx in training_idxs])
        res = filter(lambda x:  x is not None, results)
        for k, v in res:
            model.trained_on.append(k)
            predictions[k] = v

    # Create logs
    with open(f"{path}/logs.txt", "w", encoding="utf-8") as f:
        f.write(model.logs)

    # Save predictions
    save_h5(predictions, f"{path}/predictions.h5")

# Test each model. If there is only one, then use sequential, otherwise parallel
def test_all_models(models: list[Model]):
    training_idxs = [
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
    ]

    for model in models:
        test_model(model, training_idxs)

# Debug the model - only train it on first 5
def debug_model(model: Model):
    path = get_folder_directory(ROOT, model)
    pred = train(model, TrainingIndex("20 to 40", range(20, 40)), path)
    if pred is None:
        print("Encountered training error")
        return
    
    predictions = {"20 to 40": pred[1]}

    with open(f"{path}/logs.txt", "w", encoding="utf-8") as f:
        f.write(model.logs)
    
    save_h5(predictions, f"{path}/predictions.h5")
    make_plots(path, None, ["20 to 40"])
    
if __name__ == "__main__":
    test_all_models([
        SymmetricNNModel(epochs = 100),
        PoissonNNModel(epochs = 50),
        SymmetricPoissonModel(epochs=100)
    ])