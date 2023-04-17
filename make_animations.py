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
FOLDER_NAME = "./Datas/Week 3"

def main():
    processes = []
    for path in os.listdir(FOLDER_NAME):
        p = Process(target=make_plots, args=(f"{FOLDER_NAME}/{path}",))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()