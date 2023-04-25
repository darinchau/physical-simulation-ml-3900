# Run this if you want to generate all animations
# Takes a long time so we use multiprocessing but the following is essentially identical to this code

# for path in os.listdir("./Datas/Week 3"):
#     make_plots(f"./Datas/Week 3/{path}")


import os
from load import make_plots
from multiprocessing import Process

# This suppresses the warnings when doing the log plots because theres gotta be a zero somewhere :)
import warnings
warnings.filterwarnings("ignore")

# The folder that contains the generated data
FOLDER_NAME = "./Datas/Week 4"

# The list of training results to include in the error plots
# We will still make all the animations regardless of this list
RESULTS_TO_INCLUDE = [
    "First 5", 
    # "First 20", 
    # "First 40", 
    # "First 90", 
    # "20 to 40", 
    # "40 to 60"
]

def main():
    processes = []
    for path in os.listdir(FOLDER_NAME):
        p = Process(target=make_plots, args=(f"{FOLDER_NAME}/{path}", None, RESULTS_TO_INCLUDE))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
