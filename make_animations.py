# Run this if you want to generate all animations
# Takes a long time so we use multiprocessing but the following is essentially identical to this code

# for path in os.listdir("./Datas/Week 3"):
#     make_plots(f"./Datas/Week 3/{path}")

import os
from anim import make_plots
from multiprocessing import Process, Pool
import time

# This suppresses the warnings when doing the log plots because theres gotta be a zero somewhere :)
import warnings
warnings.filterwarnings("ignore")

# The folder that contains the generated data
FOLDER_NAME = "./Datas/Week 6"

# The list of training results to include in the error plots
# We will still make all the animations regardless of this list
RESULTS_TO_INCLUDE = [
    "First 5", 
    "First 20", 
    "First 40", 
    "First 90", 
    "20 to 40", 
    "40 to 60"
]

def make_animations(path: str):
    model_name = path.split("/")[-1]
    for file_name in os.listdir(path):
        if file_name.endswith('.gif'):
            print(f"Skipping making animations for {model_name}")
            return
    print(f"Making animations for {model_name}")
    make_plots(path, None, RESULTS_TO_INCLUDE)
    print(f"Finished making animations for {model_name}")

def main():
    paths = [f"{FOLDER_NAME}/{path}" for path in os.listdir(FOLDER_NAME)]
    t1 = time.time()
    with Pool(processes=8) as pool:
        pool.starmap(make_animations, [(path,) for path in paths])
    print(f"Finished making all animation in {time.time() - t1:.4f} seconds")

if __name__ == "__main__":
    main()
