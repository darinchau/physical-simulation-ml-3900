import numpy as np
from models import *
from load import *
from dataclasses import dataclass
import multiprocessing as mp
import torch
from anim import AnimationMaker, DataVisualizer, log_diff
from scipy.optimize import minimize

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
def train(model: Model, training_idxs: list[TrainingIndex], root: str):
    vg = Dataset(np.arange(101).reshape(101, 1) / 100 * 0.0075)
    potential = Dataset(load_elec_potential())
    edensity = Dataset(load_e_density())
    space_charge = Dataset(load_space_charge())

    predictions = {}
    
    for idxs in training_idxs:
        # Refresh the model
        model.refresh()

        # Split out the training data
        xtrain, xtest = vg.split_at(idxs)
        ytrain, ytest = potential.split_at(idxs)
        edtrain, edtest = edensity.split_at(idxs)
        sctrain, sctest = space_charge.split_at(idxs)
        
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
            continue

        # Test the model
        ypred = model.predict(vg).to_tensor().cpu().numpy().reshape(101, 129, 17)

        # Save the prediction
        predictions[idxs.name] = np.array(ypred)

        # Save the model
        model.save(root)

    return predictions

# Puts a model to the test with the given training indices
def test_model(model: Model, training_idxs: list[TrainingIndex]):
    # Create folder
    path = get_folder_directory(ROOT, model)

    # Train the model
    predictions = train(model, training_idxs, path)

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

    if len(models) == 1:
        test_model(models[0], training_idxs)
        return
    
    with mp.Pool(processes = 4) as pool:
        pool.starmap(test_model, [(model, training_idxs) for model in models])
    
if __name__ == "__main__":
    anim = AnimationMaker()
    target = load_elec_potential()
    x_spacing, y_spacing = load_spacing()

    # Flips the array without linear interpolation
    def flip(total):
        target_flipped = np.zeros_like(target)
        for j in range(129):
            x = total - x_spacing[j]
            col = np.abs(x - x_spacing).argmin()
            target_flipped[:, j, :] = target[:, col, :]
        return target_flipped
    
    target_flipped = flip(0.077964)

    # Flip with linear interpolation
    def flip_lerp(total):
        target_flipped = np.zeros_like(target)
        for j in range(129):
            x = total - x_spacing[j]
            col = np.abs(x - x_spacing).argmin()
            col1 = np.searchsorted(x_spacing, x, side='right') - 1
            col2 = col1 + 1
            if col2 == 129:
                target_flipped[:, j, :] = target[:, 128, :]
            else:
                weighting = (x_spacing[col2] - x)/(x_spacing[col2] - x_spacing[col1])
                target_flipped[:, j, :] = weighting * target[:, col1, :] + (1-weighting) * target[:, col2, :]
        return target_flipped
    
    a = 999
    best_total = 999999
    for i in range(77000, 83000):
        total = i/1000000
        if i%1000 == 0:
            print(total)
        diff = np.sum(np.abs(flip_lerp(total) - target))
        if diff < best_total:
            best_total = diff
            a = total
            print(f"Found new best total: {best_total} at a = {a}")
    print(a)
    target_lerp_flip = flip_lerp(a)

    # dv = DataVisualizer()
    # dv.add_data(target, "Original", 3)
    # dv.add_data(target_flipped, "Flipped")
    # dv.add_data(target_literal_flip, "Literal flipped")
    # dv.show()

    anim.add_data(np.abs(target - target_flipped), "Flip difference")
    anim.add_data(log_diff(target, target_flipped), "Flip difference log", vmin = -8)
    print(f"Total difference for flip = {np.sum(np.abs(target_flipped - target))}")
    
    anim.add_data(np.abs(target - target_lerp_flip), "Flip difference lerp")
    anim.add_data(log_diff(target, target_lerp_flip), "Flip difference lerp log", vmin = -8)
    print(f"Total difference for lerp flip = {np.sum(np.abs(target_lerp_flip - target))}")
    
    anim.add_text([f"Frame {i}" for i in range(101)])
    anim.save("flipped difference.gif")
