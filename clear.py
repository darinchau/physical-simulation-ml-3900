# Run this if you want to clear up all animations and whatnots

import os

PATH = "./Datas/Week 7"

if __name__ == "__main__":
    for foldername, subfolders, filenames in os.walk(PATH):
        for filename in filenames:
            if filename.endswith('.png') or filename.endswith('.gif'):
                os.remove(os.path.join(foldername, filename))
