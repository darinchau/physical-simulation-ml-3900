# A module to store all the driver code for training

import numpy as np
from models import *
from load import *
import multiprocessing as mp
from anim import make_plots
from multiprocessing import Pool
from models_base import TrainingIndex

def train(mf: ModelFactory, idx: TrainingIndex, root: str):
    # Get the datas
    print(f"Starting training for {mf.name} on {idx.name}")
    vg = Dataset(np.arange(101).reshape(101, 1) / 100 * 0.75)
    potential = Dataset(load_elec_potential())
    edensity = Dataset(load_e_density())
    space_charge = Dataset(load_space_charge())

    # Refresh the model
    model = mf.get_new(idx.name)

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
    model.inform(root, "path")
    model.inform(idx.name, "inputname")

    # Train the model
    try:
        model.fit(xtrain, ytrain)
        ypred = model.predict(vg).to_tensor().cpu().numpy().reshape(101, 129, 17)
    except TrainingError as e:
        print(f"Aborting training for {mf.name} on {idx.name} - {e}")
        return None
    
    # Save the model
    model.save(root, idx.name)

    # Print logs
    print(f"Finished training {model.name} with {idx.name}")

    return idx.name, np.array(ypred)

TRAINING_IDXS = {
    "First 5": TrainingIndex("First 5", 0, 5),
    "First 20": TrainingIndex("First 20", 0, 20),
    "First 30": TrainingIndex("First 30", 0, 30),
    "First 40": TrainingIndex("First 40", 0, 40),
    "First 60": TrainingIndex("First 60", 0, 60),
    "First 75": TrainingIndex("First 75", 0, 75),
    "First 90": TrainingIndex("First 90", 0, 90),
    "15 to 45": TrainingIndex("15 to 45", 15, 45),
    "20 to 40": TrainingIndex("20 to 40", 20, 40),
    "40 to 60": TrainingIndex("40 to 60", 40, 60),
    "25 to 35": TrainingIndex("25 to 35", 25, 35),
    "20 to 50": TrainingIndex("20 to 50", 20, 50),
    "30 to 50": TrainingIndex("30 to 50", 30, 50),
    "29 and 30 and 31": TrainingIndex("29 and 30 and 31", 29, 32)
}

class Trainer:
    def __init__(self, root: str, train_fn = None):
        if train_fn is None:
            self.train_fn = train
        else:
            self.train_fn = train_fn
        self.root = root
    
    # Puts a model to the test with the given training indices
    def test_model(self, mf: ModelFactory, training_idxs: list[TrainingIndex], force_sequential: bool = False):        
        # Create folder
        path = get_folder_directory(self.root, mf)
        predictions = {}

        # Train model for each index
        if mf.threads == 1 or force_sequential:
            results = []
            for idx in training_idxs:
                results.append(self.train_fn(mf, idx, path))
        else:
            with Pool(processes=mf.threads) as pool:
                results = pool.starmap(self.train_fn, [(mf, idx, path) for idx in training_idxs])
        
        res = filter(lambda x:  x is not None, results)
        for k, v in res:
            mf.trained_on.append(k)
            predictions[k] = v

        # Create logs
        with open(f"{path}/logs.txt", "w", encoding="utf-8") as f:
            f.write(mf.logs)

        # Save predictions
        save_h5(predictions, f"{path}/predictions.h5")

    # Test each model. If there is only one, then use sequential, otherwise parallel
    def test_all_models(self, models: list[ModelFactory], force_sequential: bool = False):
        training_idxs = [v for _, v in TRAINING_IDXS.items()]
        for model in models:
            self.test_model(model, training_idxs, force_sequential=force_sequential)

    # Debug the model - only train it on first 5
    def debug_model(self, model: ModelFactory, training_idx: str = "20 to 40"):
        if training_idx not in TRAINING_IDXS:
            raise KeyError(f"Unknown training index: {training_idx}")
        idx = TRAINING_IDXS[training_idx]
        path = get_folder_directory(self.root, model)
        pred = self.train_fn(model, idx, path)
        if pred is None:
            print("Encountered training error")
            return
        
        predictions = {training_idx: pred[1]}

        with open(f"{path}/logs.txt", "w", encoding="utf-8") as f:
            f.write(model.logs)
        
        save_h5(predictions, f"{path}/predictions.h5")
        make_plots(path, None, [training_idx])
    