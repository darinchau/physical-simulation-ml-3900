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

def straight_line_score(y, algorithm = 'linear') -> float:
    """Measures how close the given plots resemble a linear relationship. This measurement is invariant to scaling"""
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
