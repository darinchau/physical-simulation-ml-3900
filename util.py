import gc
import torch

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