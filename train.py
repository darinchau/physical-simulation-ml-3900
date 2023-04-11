from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression as Linear, RidgeCV, MultiTaskElasticNetCV, MultiTaskLassoCV, ElasticNetCV, LassoCV, BayesianRidge, OrthogonalMatchingPursuitCV, SGDRegressor as SGD, PassiveAggressiveRegressor as PassiveAggressive
from sklearn.tree import DecisionTreeRegressor as DecisionTree
import numpy as np
from abc import abstractmethod as virtual
from load import index_exclude, index_include
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary
import time
import matplotlib.pyplot as plt

# Filter all warnings
import warnings
warnings.filterwarnings('ignore')

class Regressor:
    @property
    @virtual
    def model_name(self):
        raise NotImplementedError

    @property
    def model(self):
        if not hasattr(self, "_model"):
            raise AttributeError("Regression model has not been trained yet")
        return self._model

    def set_path(self, path):
        # Sets the path to save whatever produced inside the model
        self._path = path

    # Sets the name of the input so that stuff is saved with names nicely
    def set_input_name(self, name):
        self._input_name = name

    @property
    def path(self):
        if not hasattr(self, "_path"):
            return ""
        return self._path

    @property
    def trained(self):
        return hasattr(self, "_model")

    def predict(self, xtest):
        return self.model.predict(xtest)

    @property
    def train_info(self):
        st = f"Trained on {self.model_name} with:"
        for k, v in self.__dict__.items():
            if k[0] == "_":
                continue
            st += f"\n{k} = {str(v)}"
        return st

    @virtual
    def fit_model(self, xtrain, ytrain):
        raise NotImplementedError

    def preprocess(self, inputs, raw_data, training_idx):
        # Reshape data because gaussian process expects one dimensional output only
        num_data = raw_data.shape[0]

        # Make a copy
        data = np.array(raw_data)
        data = data.reshape((num_data, -1))

        # Create and split training data and testing data via index
        idxs = np.arange(num_data)

        train_idx, test_idx = index_include(idxs, training_idx), index_exclude(idxs, training_idx)

        # Create the data
        xtrain, xtest = inputs[train_idx], inputs[test_idx]
        ytrain, ytest = data[train_idx], data[test_idx]

        if len(xtrain.shape) == 1:
            xtrain = xtrain.reshape(-1, 1)
            xtest = xtest.reshape(-1, 1)

        # Store the number of inputs and outputs just in case we need them
        self._num_inputs: int = xtrain.shape[1]
        self._num_outputs: int = ytrain.shape[1]

        return xtrain, xtest, ytrain, ytest

    def calculate_error(self, xtest, ytest, skip_error, verbose):
        # Calculate the error
        if skip_error:
            return

        # Calculate error
        ypred = self.predict(xtest)

        # Calculate the mean square error
        mse = np.mean((ytest - ypred)**2)
        worst = np.max(np.abs(ytest - ypred))

        if verbose:
            print(f"{self.model_name} - MSE: {mse}, worse: {worst}")

        return mse, worst

    # Preprocess, fit data, and calculate error
    def fit(self, inputs, raw_data, training_idx, verbose = False, skip_error = False):
        # Split data
        xtrain, xtest, ytrain, ytest = self.preprocess(inputs, raw_data, training_idx)

        # Train the model
        np.random.seed(12345)
        model = self.fit_model(xtrain, ytrain)

        self._model = model

        # Calculate and return error
        err = self.calculate_error(xtest, ytest, skip_error, verbose)

        return err

class GaussianRegression(Regressor):
    def fit_model(self, xtrain, ytrain):
        # Define the kernel function
        kernel = RBF(length_scale=1.0)

        # Define the Gaussian Process Regression model
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

        # Train the model on the training data
        model.fit(xtrain, ytrain)

        return model

    @property
    def model_name(self):
        return "Gaussian process"

class LinearRegression(Regressor):
    def fit_model(self, xtrain, ytrain):
        model =  Linear().fit(xtrain, ytrain)
        return model

    @property
    def model_name(self):
        return "Linear regression"

class RidgeCVRegression(Regressor):
    def __init__(self, cv=5):
        self.cv = cv

    def fit_model(self, xtrain, ytrain):
        # Fit the RidgeCV regression model on the training data
        model = RidgeCV(cv=self.cv).fit(xtrain, ytrain)

        # Return the trained model
        return model

    @property
    def model_name(self):
        return "RidgeCV Regression"

class MultiTaskElasticNetCVRegression(Regressor):
    def __init__(self, l1_ratio=0.5, cv=5):
        self.l1_ratio = l1_ratio
        self.cv = cv

    def fit_model(self, xtrain, ytrain):
        # Fit the MultiTaskElasticNetCV regression model on the training data
        model = MultiTaskElasticNetCV(l1_ratio=self.l1_ratio, cv=self.cv).fit(xtrain, ytrain)

        # Return the trained model
        return model

    @property
    def model_name(self):
        return "MultiTaskElasticNetCV Regression"


class MultiTaskLassoCVRegression(Regressor):
    def __init__(self, cv=5):
        self.cv = cv

    def fit_model(self, xtrain, ytrain):
        # Fit the MultiTaskLassoCV regression model on the training data
        model = MultiTaskLassoCV(cv=self.cv).fit(xtrain, ytrain)

        # Return the trained model
        return model

    @property
    def model_name(self):
        return "MultiTaskLassoCV Regression"

class DecisionTreeRegression(Regressor):
    def __init__(self, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    def fit_model(self, xtrain, ytrain):
        return DecisionTree(criterion=self.criterion, splitter=self.splitter, max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                                      min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
                                      random_state=self.random_state, max_leaf_nodes=self.max_leaf_nodes,
                                      min_impurity_decrease=self.min_impurity_decrease).fit(xtrain, ytrain)

    @property
    def model_name(self):
        return f"Decision Tree Regressor"


#####################################################################################################################
#### The following implements a wrapper class for regressors which can only predict one single feature           ####
#### and make them predict multiple features by making them predict single features independently and separately ####
#####################################################################################################################
class MultipleRegressor(Regressor):
    # Preprocess, fit data, and calculate error
    def fit(self, inputs, raw_data, training_idx, verbose = False, skip_error = False):
        # Split data
        xtrain, xtest, ytrain, ytest = self.preprocess(inputs, raw_data, training_idx)

        # Number of predictions needed to be made
        _, num_tasks = ytest.shape

        # Train the model
        np.random.seed(12345)

        models = [None] * num_tasks

        for i in range(num_tasks):
            models[i] = self.fit_model(xtrain, ytrain[:, i])

        self._model = models

        err = self.calculate_error(xtest, ytest, skip_error, verbose)

        return err

    def predict(self, xtest):
        num_tasks = len(self._model)
        num_samples = len(xtest)

        ypred = np.zeros((num_samples, num_tasks))

        for i in range(num_tasks):
            ypred[:, i] = self._model[i].predict(xtest)

        return ypred

class BayesianRidgeRegression(MultipleRegressor):
    def __init__(self, n_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def fit_model(self, xtrain, ytrain):
        return BayesianRidge(n_iter=self.n_iter, tol=self.tol, alpha_1=self.alpha_1, alpha_2=self.alpha_2,
                             lambda_1=self.lambda_1, lambda_2=self.lambda_2).fit(xtrain, ytrain)

    @property
    def model_name(self):
        return "Bayesian Ridge"

class SGDRegression(MultipleRegressor):
    def __init__(self, loss='squared_error', penalty='l2', alpha=0.0001, l1_ratio=0.15, max_iter=1000, tol=1e-3):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit_model(self, xtrain, ytrain):
        return SGD(loss=self.loss, penalty=self.penalty, alpha=self.alpha, l1_ratio=self.l1_ratio,
                             max_iter=self.max_iter, tol=self.tol).fit(xtrain, ytrain)

    @property
    def model_name(self):
        return "SGD Regressor"

class PassiveAggressiveRegression(MultipleRegressor):
    def __init__(self, C=1.0, fit_intercept=True, max_iter=10000, tol=1e-6):
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit_model(self, xtrain, ytrain):
        return PassiveAggressive(C=self.C, fit_intercept=self.fit_intercept,
                                          max_iter=self.max_iter, tol=self.tol).fit(xtrain, ytrain)

    @property
    def model_name(self):
        return "Passive Aggressive Regressor"

##################################################################################################################
#### This defines wrapper classes around Pytorch Neural networks so we hopefully simplify things a little bit ####
##################################################################################################################
# Using this class, one must additionally define the init_net method, as well as the model name and training details
class NeuralNetworkRegressor(Regressor):
    # path: path to save file for stuff
    def __init__(self, epochs: tuple[int, int] = (100, 500), show_training_logs = False):
        self._epochs = epochs
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Batch size
        self._bs = 1
        self._show_logs = show_training_logs

    # Preprocess, fit data, and calculate error
    def fit(self, inputs, raw_data, training_idx, verbose = False, skip_error = False):
        # Set random seed for reproducible result
        np.random.seed(12345)

        # Split data
        xtrain, xtest, ytrain, ytest = self.preprocess(inputs, raw_data, training_idx)

        # Wrap everything in dataloaders to prepare for training
        # Convert numpy arrays to PyTorch tensors and move to device
        xtrain_tensor = torch.tensor(xtrain).to(self._device).float()
        xtest_tensor = torch.tensor(xtest).to(self._device).float()
        ytrain_tensor = torch.tensor(ytrain).to(self._device).float()
        ytest_tensor = torch.tensor(ytest).to(self._device).float()

        train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
        test_dataset = TensorDataset(xtest_tensor, ytest_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Begin the training
        net = self.init_net().to(self._device)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        history = {
            'train_x': [],
            'train_y': [],
            'test_x': [],
            'test_y': []
        }

        # Define the number of epochs to train
        num_epochs, num_training_per_epoch = self._epochs

        num_trains = 0

        # Train the neural network
        for epoch in range(num_epochs):
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

            if self._show_logs:
                print(f"Trained {epoch}/{num_epochs} epochs for {self.model_name} on {self._input_name}")

        # Plot the training details
        fig, ax = plt.subplots()
        ax.plot(history['train_x'], history['train_y'])
        ax.plot(history['test_x'], history['test_y'])
        ax.set_yscale('log')
        ax.legend(['Train Error', 'Test Error'])
        ax.set_title(f"Train/Test Error over epochs for trainind {self.model_name} on {self._input_name}")
        fig.savefig(f"{self.path}/{self._input_name} training details.png")

        # Skip saving the model for now
        # self.save(net)
        self._model = net

        # Calculate error
        err = self.calculate_error(xtest, ytest, skip_error=skip_error, verbose=verbose)

        return err

    def fit_model(self, xtrain, ytrain):
        raise NotImplementedError

    @virtual
    def init_net(self) -> nn.Module:
        raise NotImplementedError

    def save(self, model: nn.Module):
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(f'{self.path}/{self._input_name}.pt')

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
        l, r = 38, 38
        m = 129-l-r

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.left = nn.Sequential(
                        nn.Linear(1, l*17),
                    )
                self.middle = nn.Sequential(
                    nn.Linear(1, 10),
                    nn.Linear(10, 25),
                    nn.Linear(25, m*17)
                )
                self.right = nn.Sequential(
                        nn.Linear(1, r*17),
                    )

            def forward(self, x):
                left = self.left(x).reshape((-1, l, 17))
                mid = self.middle(x).reshape((-1, m, 17))
                right = self.right(x).reshape((-1, r, 17))
                res = torch.cat([left, mid, right], dim = 1).reshape((-1, 1, 2193))
                return res

        return Net()

    @property
    def model_name(self):
        return "Week 2 Net 2"

class Week2Net3(NeuralNetworkRegressor):
    def init_net(self) -> nn.Module:
        l, r = 38, 38
        m = 129-l-r

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.left = nn.Sequential(
                        nn.Linear(1, 10),
                        nn.Linear(10, l*17),
                    )
                self.middle = nn.Sequential(
                    nn.Linear(1, 10),
                    nn.Linear(10, 25),
                    nn.Linear(25, m*17)
                )
                self.right = nn.Sequential(
                        nn.Linear(1, 10),
                        nn.Linear(10, r*17),
                    )

            def forward(self, x):
                left = self.left(x).reshape((-1, l, 17))
                mid = self.middle(x).reshape((-1, m, 17))
                right = self.right(x).reshape((-1, r, 17))
                res = torch.cat([left, mid, right], dim = 1).reshape((-1, 1, 2193))
                return res

        return Net()

    @property
    def model_name(self):
        return "Week 2 Net 3"


# Import antics
__all__ = [
    "Regressor",
    "DecisionTreeRegression",
    "RidgeCVRegression",
    "GaussianRegression",
    "SGDRegression",
    "PassiveAggressiveRegression",
    "LinearRegression",
    "MultiTaskLassoCVRegression",
    "MultiTaskElasticNetCVRegression",
    "BayesianRidgeRegression",
    "Week1Net1",
    "Week2Net1",
    "Week2Net2",
    "Week2Net3"
]