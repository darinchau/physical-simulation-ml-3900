# Run this if you want to clear up all animations and whatnots

import os

if __name__ == "__main__":
    for foldername, subfolders, filenames in os.walk("./Datas/Week 3"):
        for filename in filenames:
            if filename.endswith('.png') or filename.endswith('.gif'):
                os.remove(os.path.join(foldername, filename))
