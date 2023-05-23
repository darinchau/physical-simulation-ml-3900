from __future__ import annotations
from collections.abc import Iterator
import gc
import torch
from numpy.typing import NDArray
import numpy as np
from sklearn.linear_model import TheilSenRegressor, LinearRegression

class Progress:
    def __init__(self, pad = 200):
        self.pad = pad
    
    def rint(self, content: str):
        print(content.ljust(self.pad), end = '\r')
        self.pad = max(self.pad, len(content) + 1)

# A helper class to monitor cuda usage for debugging
# Use this with the debugger to create an ad hoc cuda memory watcher in profile txt
class CudaMonitor:
    # Property flag forces things to save everytime a line of code gets run in the debugger
    @property
    def memory(self):
        print("Logging memory")
        s = []
        num_tensors = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    # Total numbers
                    total = 1
                    for i in obj.shape:
                        total *= i
                    s.append((total, f"Tensor: {type(obj)}, size: {obj.size()}, shape: {obj.shape}"))
                    num_tensors += 1
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                pass
        s = [x[1] for x in sorted(s, key = lambda a: a[0], reverse = True)]
        with open("profile.txt", 'w') as f:
            f.write(f"Memory allocated: {torch.cuda.memory_allocated()}\n")
            f.write(f"Max memory allocated: {torch.cuda.max_memory_allocated()}\n")
            for y in s:
                f.write(y)
                f.write("\n")
        return f"Logged {num_tensors} tensors at profile.txt"
    
    def clear(self):
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    del obj
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                pass

def straight_line_model(algorithm):
    algorithms = {
        'linear': LinearRegression(),
        'theilsen': TheilSenRegressor()
    }

    try:
        model = algorithms[algorithm]
        return model
    except KeyError:
        e = f"Unknown algorithm ({algorithm}). Algorithm must be one of: "
        e += ", ".join(algorithms.keys())
        raise ValueError(e)

def array(d: torch.Tensor | NDArray) -> NDArray:
    """Turns a tensor or numpy array to numpy arrau"""
    if isinstance(d, torch.Tensor):
        with torch.no_grad():
            return d.cpu().numpy()
    return np.array(d)

def straight_line_score_normalizing(y, algorithm = 'linear') -> float:
    """Measures how close the given plots resemble a linear relationship. This measurement is invariant to scaling but as a result tiny errors can result in big differences"""
    len_y, = y.shape
    y = array(y)

    # Early exit if horizontal y
    if np.min(y) == np.max(y):
        return 1
    
    y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
    x = np.linspace(0, 1, len_y, endpoint=True).reshape(-1, 1)
    model = straight_line_model(algorithm).fit(x, y_normalized)
    yp = model.predict(x)
    r_score = 1 - np.sqrt(np.mean((yp - y_normalized) ** 2))
    return float(r_score)

def straight_line_score(y, algorithm = 'linear') -> float:
    """Measures how close the given plots resemble a linear relationship. This is calculated as 1/(1 + RMSE)"""
    len_y, = y.shape
    y = array(y)

    # Early exit if horizontal y
    if np.min(y) == np.max(y):
        return 1
    
    x = np.linspace(0, 1, len_y, endpoint=True).reshape(-1, 1)
    model = straight_line_model(algorithm).fit(x, y)
    yp = model.predict(x)
    r_score = np.sqrt(np.mean((yp - y) ** 2))
    return float(1/(1+r_score))

# A wrapper class for training indices. Originally we make this inherit from slice so we can directly index them into tensors
# But slice is a final object so we use a workaround to subclass from tuple instead
# But this is hacky workaround dont do this
class TrainingIndex(tuple):
    def __new__(cls, name: str, start: int, stop: int, /):
        # We use a list of a single range because this code works
        # import torch
        # a = torch.arange(16).reshape(4,4)
        # a[((1, 2, 3),)]
        # tensor([[ 4,  5,  6,  7],
        #         [ 8,  9, 10, 11],
        #         [12, 13, 14, 15]])
        self = super().__new__(cls, (tuple(range(start, stop)),))
        self.name = name
        return self


# For easy access of training indices
class TrainingIndexContainer:
    def __init__(self, ls) -> None:
        self.ls = ls

    def __iter__(self):
        return iter(self.ls)
    
    def __getitem__(self, a):
        if isinstance(a, int):
            return self.ls[a]
        
        if isinstance(a, str):
            if a.startswith("First "):
                return TrainingIndex(a, 0, int(a.split("First ")[1]))
            if a.count("to"):
                return TrainingIndex(a, int(a.split("to")[0]), int(a.split("to")[1]))
            
        raise TypeError


TRAINING_IDXS = TrainingIndexContainer([
            TrainingIndex("First 5", 0, 5),
            TrainingIndex("First 20", 0, 20),
            TrainingIndex("First 30", 0, 30),
            TrainingIndex("First 40", 0, 40),
            TrainingIndex("First 60", 0, 60),
            TrainingIndex("First 75", 0, 75),
            TrainingIndex("First 90", 0, 90),
            TrainingIndex("15 to 45", 15, 45),
            TrainingIndex("20 to 40", 20, 40),
            TrainingIndex("40 to 60", 40, 60),
            TrainingIndex("25 to 35", 25, 35),
            TrainingIndex("20 to 50", 20, 50),
            TrainingIndex("30 to 50", 30, 50),
            TrainingIndex("29 to 32", 29, 32)
        ])