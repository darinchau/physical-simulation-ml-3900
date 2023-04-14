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
from numpy import log
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
    def fit_model(self, xtrain, ytrain, xtest, ytest):
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
        model = self.fit_model(xtrain, ytrain, xtest, ytest)

        self._model = model

        # Calculate and return error
        err = self.calculate_error(xtest, ytest, skip_error, verbose)

        return err

class GaussianRegression(Regressor):
    def fit_model(self, xtrain, ytrain, xtest, ytest):
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
    def fit_model(self, xtrain, ytrain, xtest, ytest):
        model =  Linear().fit(xtrain, ytrain)
        return model

    @property
    def model_name(self):
        return "Linear regression"

class RidgeCVRegression(Regressor):
    def __init__(self, cv=5):
        self.cv = cv

    def fit_model(self, xtrain, ytrain, xtest, ytest):
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

    def fit_model(self, xtrain, ytrain, xtest, ytest):
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

    def fit_model(self, xtrain, ytrain, xtest, ytest):
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

    def fit_model(self, xtrain, ytrain, xtest, ytest):
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

    def fit_model(self, xtrain, ytrain, xtest, ytest):
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

    def fit_model(self, xtrain, ytrain, xtest, ytest):
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

    def fit_model(self, xtrain, ytrain, xtest, ytest):
        return PassiveAggressive(C=self.C, fit_intercept=self.fit_intercept,
                                          max_iter=self.max_iter, tol=self.tol).fit(xtrain, ytrain)

    @property
    def model_name(self):
        return "Passive Aggressive Regressor"


### The following implements neural networks in a better way (than week 2)
### The difference being we hide the training code in a separate class
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

# With this wrapper class we can use Neural Net Model like a normal model
class NeuralNetModel(nn.Module):
    @virtual
    # This will be called exactly once before training
    def init_net(self, input_size, output_size):
        raise NotImplementedError

    @virtual
    # This will be called to predict stuff
    def predict_(self, xtest):
        raise NotImplementedError

    # The actual predict will be called with a wrap to convert between tensor and numpy
    def predict(self, xtest):
        with torch.no_grad():
            xtest_tensor = torch.tensor(xtest).to(self._device).float()
            result_ = self.predict_(xtest_tensor)
        return result_.cpu().numpy()

    def __init__(self):
        super(NeuralNetModel, self).__init__()
        self.trained = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.predict_(x)

    # A neural net has to have 2 things: fit (using xtrain and ytrain) and predict (using xtest)
    def fit(self, xtrain, ytrain, xtest, ytest,
            num_epochs_ = (100, 100),
            model_name = "",
            input_name = "",
            path = "./",
            save_model = True,
            show_logs = True):

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

        # Initialize the net (the layers)
        input_size = xtrain.shape[1]
        output_size = ytrain.shape[1]
        self.init_net(input_size, output_size)

        # Start training
        net = self.to(self._device)
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
        num_epochs, num_training_per_epoch = num_epochs_

        num_trains = 0
        epoch = 0

        # Train the neural network
        while True:
            # Train N times for each test
            for _ in range(num_training_per_epoch):
                train_loss = 0.0
                for _, (data, target) in enumerate(train_dataloader):
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
            if show_logs:
                print(f"Trained {epoch}/{num_epochs} epochs for {model_name} on {input_name}")

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
        ax.set_title(f"Train/Test Error over epochs for training {model_name} on {input_name}")
        fig.savefig(f"{path}/{input_name} training details.png")

        # Save the model if needed
        if save_model:
            model_scripted = torch.jit.script(net) # Export to TorchScript
            model_scripted.save(f'{path}/{input_name}_{num_epochs}epochs.pt')

        # Set a flag
        self.trained = True

        return self

# We can define custom models and regressors for said models minimally
class Week3Net1(NeuralNetModel):
    def init_net(self, input_size, output_size):
        self.fc = nn.Linear(input_size, output_size)

    def predict_(self, xtest):
        return self.fc(xtest)

# A wrapper class for Week 3 net 1
class SimpleNNRegressor(Regressor):
    def fit_model(self, xtrain, ytrain, xtest, ytest):
        return Week3Net1().fit(xtrain, ytrain, xtest, ytest,
                               model_name = self.model_name,
                               input_name = self._input_name,
                               path = self.path)

    @property
    def model_name(self):
        return "Week 3 Neural net 1"

# Now with this abstraction in mind we can build much more convoluted stuff
# Idea 1: Use Gaussian and Linear mix because linear performs quite well on the sides
# but Gaussian performs quite well in the middle
class GaussianLinearRegression(Regressor):
    def fit_model(self, xtrain, ytrain, xtest, ytest):
        ytrain = ytrain.reshape((-1, 129, 17))
        yleft = ytrain[:, :40, :].reshape((-1, 40*17))
        ymiddle = ytrain[:, 40:89, :].reshape((-1, 49*17))
        yright = ytrain[:, 89:, :].reshape((-1, 40*17))

        left_model = Linear().fit(xtrain, yleft)
        right_model = Linear().fit(xtrain, yright)

        # Define the kernel function
        kernel = RBF(length_scale=1.0)
        # Define the Gaussian Process Regression model
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10).fit(xtrain, ymiddle)

        return (left_model, model, right_model)

    def predict(self, xtest):
        l = self._model[0].predict(xtest).reshape((-1, 40, 17))
        m = self._model[1].predict(xtest).reshape((-1, 49, 17))
        r = self._model[2].predict(xtest).reshape((-1, 40, 17))

        y = np.concatenate([l, m, r], axis = 1)

        y = y.reshape((-1, 129*17))

        return y

    @property
    def model_name(self):
        return "Gaussian Linear Hybrid 1"

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
    "SimpleNNRegressor",
    "GaussianLinearRegression"
]