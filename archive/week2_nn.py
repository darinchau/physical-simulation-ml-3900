import numpy as np
from abc import abstractmethod as virtual
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary
import time
import matplotlib.pyplot as plt
from math import log
from train import Regressor

# Filter all warnings
import warnings
warnings.filterwarnings('ignore')


##################################################################################################################
#### This defines wrapper classes around Pytorch Neural networks so we hopefully simplify things a little bit ####
##################################################################################################################

# Helper function to determine whether to stop training or not
# Stop training if derivative says that the function is flat, or train/test error deviates too much
USE_LAST_N = 5
def should_exit_early(train_last_, test_last_):
    # Return some madeup value to not trigger stuff
    if len(test_last_) < USE_LAST_N:
        return False

    if len(test_last_) > USE_LAST_N:
        print("Wrong indexing!")
        return True

    # First derivative
    window_size = 3

    first_ds = []
    second_ds = []
    h = 0.01

    for i in range(USE_LAST_N - window_size + 1):
        # Estimates f'(b)
        a, b, c = test_last_[i: i + window_size]
        f_prime_b = (c - a) / (2*h)
        fpp_b = (a + c - 2*b)/h/h
        first_ds.append(f_prime_b)
        second_ds.append(fpp_b)

    fp = sum(first_ds)/(USE_LAST_N-window_size+1)
    sp = sum(second_ds)/(USE_LAST_N-window_size+1)

    if abs(fp) < 0.005 and abs(sp) < 0.005:
        return True

    for tr, te in zip(train_last_, test_last_):
        if tr > te:
            return False
        if abs(log(te) - log(tr)) < 0.75:
            return False

    return True

# Using this class, one must additionally define the init_net method, as well as the model name and training details
class NeuralNetworkRegressor(Regressor):
    # path: path to save file for stuff
    # epochs: first number is number of tests to be performed, second number is number of trainings per test
    # if first number is set to -1, then we check the first and second derivarive of the last 10 test outcome to
    # determine whether we should stop training
    def __init__(self, epochs: tuple[int, int] = (100, 250), show_training_logs = False):
        self._epochs = epochs
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Batch size
        self._bs = 1
        self._show_logs = show_training_logs


    # Preprocess, fit data, and calculate error
    def fit_model(self, inputs, raw_data, training_idx, verbose = False, skip_error = False):
        # Set random seed for reproducible result
        np.random.seed(12345)

        # Split data
        xtrain, xtest, ytrain, ytest = self.preprocess(inputs, raw_data, training_idx)

        # Wrap everything in dataloaders to prepare for training
        # Convert numpy arrays to PyTorch tensors and move to device
        xtrain_tensor = torch.tensor(xtrain).to(self._device).float()
        ytrain_tensor = torch.tensor(ytrain).to(self._device).float()

        xtest_tensor = torch.tensor(xtest).to(self._device).float()
        ytest_tensor = torch.tensor(ytest).to(self._device).float()

        train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        test_dataset = TensorDataset(xtest_tensor, ytest_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Begin the training
        net = self.init_net().to(self._device)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        history = {
            'train_x': [],
            'train_y': [],
            'test_x': [],
            'test_y': [],
            'test_y_train_loss': []
        }

        # Define the number of epochs to train
        num_epochs, num_training_per_epoch = self._epochs

        num_trains = 0
        epoch = 0

        # Train the neural network
        while True:
            for _ in range(num_training_per_epoch):
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_dataloader):
                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    output = net(data)
                    loss = criterion(output, target)

                    # Backward pass
                    loss.backward()

                    # Update the parameters
                    optimizer.step()

                    # Accumulate the training loss
                    train_loss += loss.item()

                # Calculate the average training loss
                train_loss /= len(train_dataloader)

                num_trains += 1

                # Save the training and test loss for plotting
                history['train_x'].append(num_trains)
                history['train_y'].append(train_loss)

            # Evaluate the neural network on the test set
            test_loss = 0.0

            # Evaluate the model
            with torch.no_grad():
                for data, target in test_dataloader:
                    # Move the data and target to the device
                    data, target = data, target

                    # Forward pass
                    output = net(data)
                    loss = criterion(output, target)

                    # Accumulate the test loss
                    test_loss += loss.item()

            # Calculate the average test loss
            test_loss /= len(test_dataloader)

            # Save the training and test loss for plotting
            history['test_x'].append(num_trains)
            history['test_y'].append(test_loss)
            history['test_y_train_loss'].append(history['train_y'][-1])

            # Incerement number of epochs
            epoch += 1

            # Print logs
            if self._show_logs:
                print(f"Trained {epoch}/{num_epochs} epochs for {self.model_name} on {self._input_name}")

            # Use derivative checking to exit early if necessary
            # If train error drops below test error too much we also start to worry that it will overfit
            if should_exit_early(history['test_y_train_loss'][-USE_LAST_N:], history['test_y'][-USE_LAST_N:]):
                break

            # Exit if enough epochs
            if epoch >= num_epochs:
                break

        # Plot the training details
        fig, ax = plt.subplots()
        ax.plot(history['train_x'], history['train_y'])
        ax.plot(history['test_x'], history['test_y'])
        ax.set_yscale('log')
        ax.legend(['Train Error', 'Test Error'])
        ax.set_title(f"Train/Test Error over epochs for trainind {self.model_name} on {self._input_name}")
        fig.savefig(f"{self.path}/{self._input_name} training details.png")

        # Skip saving the model for now
        self.save(net, num_trains)
        self._model = net

        # Calculate error
        err = self.calculate_error(xtest, ytest, skip_error=skip_error, verbose=verbose)

        return err

    def fit(self, xtrain, ytrain):
        raise NotImplementedError

    @virtual
    def init_net(self) -> nn.Module:
        raise NotImplementedError

    def save(self, model: nn.Module, num_epochs: int):
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(f'{self.path}/{self._input_name}_{num_epochs}epochs.pt')

    def load(self, path) -> nn.Module:
        model = torch.jit.load(path)
        model.eval()
        self._model = model

    def predict(self, xtest):
        # Convert xtest to a PyTorch tensor
        xtest_tensor = torch.tensor(xtest, dtype=torch.float32)

        # Move xtest to the same device as the trained model
        xtest_tensor = xtest_tensor.to(self._device)

        # Set the model to evaluation mode
        self.model.eval()

        # Make predictions on xtest
        with torch.no_grad():
            ypred_tensor: torch.Tensor = self._model(xtest_tensor)

        # Convert the predicted tensor to a numpy array
        ypred = ypred_tensor.cpu().numpy()

        return ypred

    # This defines the training details which output the model structure to be outputted in the logs
    # Feel free to overload this
    @property
    def train_info(self):
        model = self._model
        nin, nout = self._num_inputs, self._num_outputs
        model_stat = summary(model, input_size=(self._bs, nin), verbose=0)
        return f"Trained on {self.model_name} with structure\n\n{model_stat}"

# Now we have abstracted away all the training code and we can define neural network with one nested class only
class Week1Net1(NeuralNetworkRegressor):
    def init_net(self) -> nn.Module:
        nin, nout = self._num_inputs, self._num_outputs
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc = nn.Sequential(
                        nn.Linear(nin, 50),
                        nn.Linear(50, nout)
                    )

            def forward(self, x):
                x = self.fc(x)
                return x

        return Net()

    @property
    def model_name(self):
        return "Week 1 Net 1"

class Week2Net1(NeuralNetworkRegressor):
    def init_net(self) -> nn.Module:
        nin, nout = self._num_inputs, self._num_outputs
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc = nn.Sequential(
                        nn.Linear(nin, nout),
                    )

            def forward(self, x):
                x = self.fc(x)
                return x

        return Net()

    @property
    def model_name(self):
        return "Week 2 Net 1"

# The outer regions seems pretty constant how about we put more computing power in the middle
# TO FUTURE ME: This does not generalize to other datas with different shapes, unlike week1net1 and week2net1
class Week2Net2(NeuralNetworkRegressor):
    def init_net(self) -> nn.Module:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.left = nn.Sequential(
                        nn.Linear(1, 38*17),
                    )
                self.middle = nn.Sequential(
                    nn.Linear(1, 10),
                    nn.Linear(10, 25),
                    nn.Linear(25, 53*17)
                )
                self.right = nn.Sequential(
                        nn.Linear(1, 38*17),
                    )

            def forward(self, x):
                left = self.left(x).reshape((-1, 38, 17))
                mid = self.middle(x).reshape((-1, 53, 17))
                right = self.right(x).reshape((-1, 38, 17))
                res = torch.cat([left, mid, right], dim = 1).reshape((-1, 1, 2193))
                return res

        return Net()

    @property
    def model_name(self):
        return "Week 2 Net 2"

class Week2Net3(NeuralNetworkRegressor):
    def init_net(self) -> nn.Module:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.left = nn.Sequential(
                        nn.Linear(1, 10),
                        nn.Linear(10, 38*17),
                    )
                self.middle = nn.Sequential(
                    nn.Linear(1, 10),
                    nn.Linear(10, 25),
                    nn.Linear(25, 53*17)
                )
                self.right = nn.Sequential(
                        nn.Linear(1, 10),
                        nn.Linear(10, 38*17),
                    )

            def forward(self, x):
                left = self.left(x).reshape((-1, 38, 17))
                mid = self.middle(x).reshape((-1, 53, 17))
                right = self.right(x).reshape((-1, 38, 17))
                res = torch.cat([left, mid, right], dim = 1).reshape((-1, 1, 2193))
                return res

        return Net()

    @property
    def model_name(self):
        return "Week 2 Net 3"