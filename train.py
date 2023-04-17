from __future__ import annotations
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
import re
import math

# Filter all warnings
import warnings
warnings.filterwarnings('ignore')

class RegressorFitError(Exception):
    pass

# Returns true if the pattern says the number of splits is ass
def too_many_split(e: ValueError):
    st = e.args[0]
    pattern = r"^Cannot have number of splits n_splits=[0-9]* greater than the number of samples: n_samples=[0-9]*.$"
    return bool(re.match(pattern, st))

class Regressor:
    @property
    @virtual
    def model_name(self):
        raise NotImplementedError

    @virtual
    # Takes in xtrain and ytrain and outputs the model. The outputted model will be saved to a self.model attribute
    def fit(self, xtrain, ytrain):
        raise NotImplementedError

    @property
    def model(self):
        if not hasattr(self, "_model"):
            raise AttributeError("Regression model has not been trained yet")
        return self._model

    # Sets the path to save whatever produced inside the model
    def set_path(self, path):
        self._path = path

    # Sets the name of the input so that stuff is saved with names nicely
    def set_input_name(self, name):
        self._input_name = name

    @property
    def can_use_electron_density(self):
        return True

    @property
    def model_structure(self):
        # A property for model structure
        if not hasattr(self, "_model_structure"):
            self._model_structure = []
        return self._model_structure

    # If you use a neural network and you need to register the network with the prediction data,
    # Net initialization will be called here
    def register_net(self, net: NeuralNetwork):
        input_size = self._xtest.shape[1]
        net.init_net(input_size)
        net.register_test_data(self._xtest, self._ytest)
        self.model_structure.append(f"{summary(net, (1, input_size), verbose=0)}")
        return net

    @property
    def path(self):
        if not hasattr(self, "_path"):
            return ""
        return self._path

    def predict(self, xtest):
        return self.model.predict(xtest)

    @property
    def train_info(self):
        st = f"Trained on {self.model_name} with:"
        for k, v in self.__dict__.items():
            if k[0] == "_":
                continue
            st += f"\n{k} = {str(v)}"
        structures = "\n\n".join([str(x) for x in self.model_structure])
        st += structures
        return st

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
    def fit_model(self, inputs, raw_data, training_idx, verbose = False, skip_error = False):
        # Split data
        xtrain, xtest, ytrain, ytest = self.preprocess(inputs, raw_data, training_idx)
        self._xtest = xtest
        self._ytest = ytest

        # Train the model
        np.random.seed(12345)
        try:
            model = self.fit(xtrain, ytrain)
        except ValueError as e:
            if too_many_split(e):
                raise RegressorFitError()
            else:
                raise e

        self._model = model

class GaussianRegression(Regressor):
    def fit(self, xtrain, ytrain):
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
    def fit(self, xtrain, ytrain):
        model =  Linear().fit(xtrain, ytrain)
        return model

    @property
    def model_name(self):
        return "Linear regression"

class RidgeCVRegression(Regressor):
    def __init__(self, cv=5):
        self.cv = cv

    def fit(self, xtrain, ytrain):
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

    def fit(self, xtrain, ytrain):
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

    def fit(self, xtrain, ytrain):
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

    def fit(self, xtrain, ytrain):
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
    def fit_model(self, inputs, raw_data, training_idx, verbose = False, skip_error = False):
        # Split data
        xtrain, xtest, ytrain, ytest = self.preprocess(inputs, raw_data, training_idx)

        # Number of predictions needed to be made
        _, num_tasks = ytest.shape

        # Train the model
        np.random.seed(12345)

        models = [None] * num_tasks

        for i in range(num_tasks):
            models[i] = self.fit(xtrain, ytrain[:, i])

        self._model = models

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

    def fit(self, xtrain, ytrain):
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

    def fit(self, xtrain, ytrain):
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

    def fit(self, xtrain, ytrain):
        return PassiveAggressive(C=self.C, fit_intercept=self.fit_intercept,
                                          max_iter=self.max_iter, tol=self.tol).fit(xtrain, ytrain)

    @property
    def model_name(self):
        return "Passive Aggressive Regressor"

#########################################################################################
#### This is the third iteration of the neural network class lol let's hope it works ####
#########################################################################################

# The design philosophy is based on the following:
# - The neural network regressor should contain "fit" and "predict" methods
# - The neural network should be straightforward enough to implement, almost frictionless from pytorch
# - Then this model can be used like a regular model inside a regressor without much hassle
# - We should not compromise to allow for test data access in fit method
class NeuralNetwork(nn.Module):
    # Now turns out this is a special wrapper around the nn module
    @virtual
    def init_net(self, input_size):
        # This will be called exactly once before training
        # Initialize everything here
        raise NotImplementedError

    @virtual
    def forward(self, x):
        # Takes in an x and use your initialized model to get a y
        raise NotImplementedError

    def __init__(self) -> None:
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self._device

    def predict(self, x):
        x = torch.tensor(x).to(self.device).float()
        with torch.no_grad():
            result = self.forward(x)
        return result.cpu().numpy()

    def register_test_data(self, xtest, ytest):
        xtest_tensor = torch.tensor(xtest).to(self.device).float()
        ytest_tensor = torch.tensor(ytest).to(self.device).float()
        test_dataset = TensorDataset(xtest_tensor, ytest_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        self._test_dataloader = test_dataloader
        # self.__input_size = xtest.shape[1]

    # @property
    # def summary(self):
    #     return f"{summary(self, (1, self.__input_size), verbose=0)}"

    def fit(self, xtrain, ytrain,
            epochs = 10000,
            validate_every = 100,
            model_name = "",
            input_name = "",
            path = "./",
            save_model = True,
            show_logs = True):

        # Turn the train data into a dataloader
        train_dataloader = self.make_dataloader(xtrain, ytrain)

        # Start training
        net = self.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        history = {
            'train_x': [],
            'train_y': [],
            'test_x': [],
            'test_y': [],
            'test_y_train_loss': []
        }

        for i in range(epochs):
            train_loss = self.train_net(train_dataloader, net, optimizer, criterion, history, i)

            # Skip the testing if it is not the "validate every" part
            if i % validate_every != 0:
                continue
            self.test_net(net, criterion, history, i, train_loss, epochs, show_logs)

        self.plot_history(model_name, input_name, path, history)

        if save_model:
            self.save_model(input_name, path, net)

        return self

    def train_net(self, train_dataloader, net, optimizer, criterion, history, i):
        train_loss = 0.0
        for data, target in train_dataloader:
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

            # Save the training and test loss for plotting
        history['train_x'].append(i)
        history['train_y'].append(train_loss)
        return train_loss

    def test_net(self, net, criterion, history, i, train_loss, epochs, show_logs):
        test_loss = 0.0
        with torch.no_grad():
            for data, target in self._test_dataloader:
                    # Forward pass
                output = net(data)
                loss = criterion(output, target)

                    # Accumulate the test loss
                test_loss += loss.item()

        if show_logs:
            print(f"Trained {i+1}/{epochs} epochs with train loss: {train_loss}, test loss: {test_loss}")

            # Save the training and test loss for plotting
        history['test_x'].append(i)
        history['test_y'].append(test_loss)
        history['test_y_train_loss'].append(train_loss)

    def save_model(self, input_name, path, net):
        model_scripted = torch.jit.script(net) # Export to TorchScript
        model_scripted.save(f'{path}/{input_name}epochs.pt')

    def plot_history(self, model_name, input_name, path, history):
        fig, ax = plt.subplots()
        ax.plot(history['train_x'], history['train_y'])
        ax.plot(history['test_x'], history['test_y'])
        ax.set_yscale('log')
        ax.legend(['Train Error', 'Test Error'])
        ax.set_title(f"Train/Test Error over epochs for training {model_name} on {input_name}")
        fig.savefig(f"{path}/{input_name} training details.png")

    def make_dataloader(self, xtrain, ytrain):
        xtrain_tensor = torch.tensor(xtrain).to(self.device).float()
        ytrain_tensor = torch.tensor(ytrain).to(self.device).float()
        train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        return train_dataloader

# A sample implementation of a neural network
class SimpleNet(NeuralNetwork):
    def init_net(self, input_size):
        self.fc = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.Linear(10, 2193),
        )

    def forward(self, x):
        return self.fc(x)

class SimpleNetRegression(Regressor):
    def fit(self, xtrain, ytrain):
        model = self.register_net(SimpleNet())
        return model.fit(xtrain, ytrain, model_name=self.model_name, input_name=f"First {xtrain.shape[0]}", path=self.path)

    @property
    def model_name(self):
        return "SimpleNet Regression"

#############################################################################################
#### A hybrid regressor indicates that we train different regions using different models ####
#############################################################################################

# Its main use is to do indicate a hybrid regressor
# We might add some specific functions here in the future :)
class HybridRegressor(Regressor):
    def __init__(self, region_size = (40, 49, 40)):
        self.region_size = region_size

# Now with this abstraction in mind we can build much more convoluted stuff
# Idea 1: Use Gaussian and Linear mix because linear performs quite well on the sides
# but Gaussian performs quite well in the middle
class GLH1Regression(HybridRegressor):
    def fit(self, xtrain, ytrain):
        # Outer region size
        outer_size = self.region_size[0]
        inner_size = self.region_size[1]

        yt = ytrain.reshape((-1, 129, 17))
        yleft = yt[:, :outer_size, :].reshape((-1, outer_size * 17))
        ymiddle = yt[:, outer_size:-outer_size, :].reshape((-1, inner_size * 17))
        yright = yt[:, -outer_size:, :].reshape((-1, outer_size * 17))

        left_model = Linear().fit(xtrain, yleft)
        right_model = Linear().fit(xtrain, yright)

        # Define the Gaussian Process Regression model
        model = GaussianProcessRegressor(
            kernel = RBF(length_scale=1.0),
            alpha = 1e-5,
            n_restarts_optimizer = 10
            ).fit(xtrain, ymiddle)

        return (left_model, model, right_model)

    def predict(self, xtest):
        outer_size = self.region_size[0]
        inner_size = self.region_size[1]

        l = self.model[0].predict(xtest).reshape((-1, outer_size, 17))
        m = self.model[1].predict(xtest).reshape((-1, inner_size, 17))
        r = self.model[2].predict(xtest).reshape((-1, outer_size, 17))

        y = np.concatenate([l, m, r], axis = 1)

        y = y.reshape((-1, 129*17))

        return y

    @property
    def model_name(self):
        return "Gaussian Linear Hybrid 1"

    @property
    def can_use_electron_density(self):
        return False

# Idea 1.1: Use the same linear model to predict the sides, and then also feed that data into the gaussian model
class GLH2Regression(HybridRegressor):
    def fit(self, xtrain, ytrain):
        outer_size = self.region_size[0]
        inner_size = self.region_size[1]

        # Fit the outer model
        outer_model = Linear().fit(xtrain, ytrain)

        # Take the outer part of the result and put it into x
        y_pred = outer_model.predict(xtrain).reshape((-1, 129, 17))
        # Trim it to get the outer part
        y_pred_left = y_pred[:, :outer_size, :].reshape((-1, outer_size*17))
        y_pred_right = y_pred[:, -outer_size:, :].reshape((-1, outer_size*17))
        # Merge the arrays
        x_merge = np.concatenate([xtrain, y_pred_left, y_pred_right], axis = 1)

        # Train the middle part
        yt = ytrain.reshape((-1, 129, 17))
        ymiddle = yt[:, outer_size:-outer_size, :].reshape((-1, inner_size * 17))

        # Gaussian process to train the middle
        inner_model = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=1e-5, n_restarts_optimizer=10).fit(x_merge, ymiddle)

        return (outer_model, inner_model)

    def predict(self, xtest):
        outer_size = self.region_size[0]
        inner_size = self.region_size[1]

        # Predict outer model
        y_pred = self.model[0].predict(xtest).reshape((-1, 129, 17))

        y_pred_left = y_pred[:, :outer_size, :].reshape((-1, outer_size*17))
        y_pred_right = y_pred[:, -outer_size:, :].reshape((-1, outer_size*17))
        x_merge = np.concatenate([xtest, y_pred_left, y_pred_right], axis = 1)

        # Predict inner model
        inner = self.model[1].predict(x_merge).reshape((-1, inner_size, 17))

        y_pred[:, outer_size:-outer_size, :] = inner
        y_pred = y_pred.reshape((-1, 129*17))

        return y_pred

    @property
    def model_name(self):
        return "Gaussian Linear Hybrid 2"

    @property
    def can_use_electron_density(self):
        return False

# Idea 1.2: what if we feed the entire linear model into the Gaussian model and see what happens?
class GLH3Regression(HybridRegressor):
    def fit(self, xtrain, ytrain):
        outer_size = self.region_size[0]
        inner_size = self.region_size[1]

        # Fit the outer model
        outer_model = Linear().fit(xtrain, ytrain)

        # Train the middle part
        y_pred = outer_model.predict(xtrain)
        x_merge = np.concatenate([xtrain, y_pred], axis = 1)

        ymiddle = ytrain.reshape((-1, 129, 17))[:, outer_size:-outer_size, :].reshape((-1, inner_size * 17))

        # Gaussian process to train the middle
        inner_model = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=1e-5, n_restarts_optimizer=10).fit(x_merge, ymiddle)

        return (outer_model, inner_model)

    def predict(self, xtest):
        outer_size = self.region_size[0]
        inner_size = self.region_size[1]

        y_pred = self.model[0].predict(xtest)
        x_merge = np.concatenate([xtest, y_pred], axis = 1)

        inner = self.model[1].predict(x_merge).reshape((-1, inner_size, 17))

        y_pred = y_pred.reshape((-1, 129, 17))
        y_pred[:, outer_size:-outer_size, :] = inner
        y_pred = y_pred.reshape((-1, 129 * 17))

        return y_pred

    @property
    def model_name(self):
        return "Gaussian Linear Hybrid 3"

    @property
    def can_use_electron_density(self):
        return False


# Idea 2. The linear model seems to predict the datas really well.
# Could we try to predict the potential increase part and the depletion part separately?
# In reality, the physical behavior of the material should change gradually
# A piecewise function doesnt exist per se
# So I am guessing its kinda like a relu except smoothed out a teeny tiny little bit
# and we try to use that little bit of smoothness to guess where the change might occur
# So we first look at the results really hard, and try to guess at what voltage the depletion might start to occur
# And then use said depletion voltage to guess rest of the term
class GLH4Regression(Regressor):
    def __init__(self, use_first_n_for_linear = 5):
        self.linear_component_N = use_first_n_for_linear

    def fit(self, xtrain, ytrain):
        N = self.linear_component_N
        if xtrain.shape[0] < N:
            raise RegressorFitError("Too few training data")

        linear_component = Linear().fit(xtrain[:N], ytrain[:N])

        # Make the error terms
        pred = linear_component.predict(xtrain)
        norm = np.max(np.abs(pred - ytrain))
        error = (ytrain - pred)/norm

        # Train the error terms over another model. Let's try Gaussian for now
        kernel = RBF(length_scale=1.0)

        # Define the Gaussian Process Regression model
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

        # Train the model on the training data
        model.fit(xtrain, error)

        return (linear_component, model, norm)


    def predict(self, xtest):
        (lin, err, norm) = self.model
        lin_component = lin.predict(xtest)
        err_component = err.predict(xtest)
        return err_component * norm + lin_component

    @property
    def model_name(self):
        return "Gaussian Linear Hybrid 4"

# Let's expand on idea 2 and build a regressor that retrains the second model with the last few predicted results
class TCEPNet(NeuralNetwork):
    def init_net(self, input_size):
        self.voltage = nn.Linear(1, 1)
        N = self.N = int((input_size - 1)/2193)

        self.conv1 = nn.Conv2d(in_channels=N, out_channels=2*N, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=2*N, out_channels=4*N, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=4*N, out_channels=6*N, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=6*N, out_channels=8*N, kernel_size=3, padding=0)
        self.flatten = nn.Flatten()

        # 116 is 58 * 2 - check calculations below
        final_num_features = 8*N*116 + 1

        # The final NN part
        self.nn1 = nn.Linear(final_num_features, 3000)
        self.dropout = nn.Dropout(p=0.3)
        self.nn2 = nn.Linear(3000, 2193)

    def forward(self, x):
        # Split into voltage part and error part
        x_voltage = x[:, :1]
        x_error = x[:, 1:].reshape((-1, self.N, 129, 17))

        # Voltage part
        x_voltage = self.voltage(x_voltage)

        # Error part
        # Shapes calcualted assuming batch size = 1
        x_error = self.conv1(x_error) # (1, 10, 127, 15)
        x_error = nn.functional.relu(x_error)
        x_error = self.conv2(x_error) # (1, 20, 125, 13)
        x_error = nn.functional.relu(x_error)
        x_error = self.pool(x_error) # (1, 20, 62, 6)
        x_error = self.conv3(x_error) # (1, 30, 60, 4)
        x_error = nn.functional.relu(x_error)
        x_error = self.conv4(x_error) # (1, 40, 58, 2)
        x_error = nn.functional.relu(x_error)
        x_error = self.flatten(x_error) # (1, 4640)

        # Combine both parts
        # Combine the two tensors
        x = torch.cat((x_voltage, x_error), dim=1)
        x = self.nn1(x)
        x = self.dropout(x)
        x = self.nn2(x)

        return x


# Time Convolution Error Prediction
class TCEPRegression(Regressor):
    def __init__(self, use_first_n_for_linear = 5):
        self.linear_component_N = use_first_n_for_linear

    # We do the training directly in fit_model so leave this unimplemented
    def fit(self, xtrain, ytrain):
        raise NotImplementedError

    @property
    def can_use_electron_density(self):
        return False

    def fit_model(self, inputs, raw_data, training_idx, verbose=False, skip_error=False):
        N = self.linear_component_N

        # Preprocess data
        xtrain, xtest, ytrain, ytest = self.preprocess(inputs, raw_data, training_idx)

        if xtrain.shape[0] < N:
            raise RegressorFitError("Too few training data")

        # Remerge x and y
        x = np.concatenate([xtrain, xtest], axis = 0)
        y = np.concatenate([ytrain, ytest], axis = 0)
        num_total_datas = x.shape[0]

        # Train the first model
        np.random.seed(12345)
        linear_model = Linear().fit(xtrain[:N], ytrain[:N])

        # Calculate the error terms
        linear_component = linear_model.predict(x).reshape((-1, 2193))
        normalization_term = np.max(np.abs(y - linear_component))
        error_terms = (y - linear_component)/normalization_term

        # Make the error terms with strides for the second phase of training
        # For example N = 5
        # So first data will contain errors for frame 0, 1, 2, 3, 4, and error_y will contain error for frame 5
        # We can use this piece of data to do something interesting
        error_with_strides = np.zeros((num_total_datas-N, N, 2193))
        error_y = np.zeros((num_total_datas-N, 2193))
        for i in range(N, num_total_datas):
            error_with_strides[i-N] = error_terms[i-N:i]
            error_y[i-N] = error_terms[i]

        error_with_strides = error_with_strides.reshape((-1, N*2193))

        # Preprocessing for second phase of training
        new_x = np.concatenate([x[N:], error_with_strides], axis = 1)
        new_num_training_data = xtrain.shape[0] - N + 1

        error_xtrain = new_x[:new_num_training_data]
        error_xtest = new_x[new_num_training_data:]
        error_ytrain = error_y[:new_num_training_data]
        error_ytest = error_y[new_num_training_data:]

        # Train CNN here
        tcep = TCEPNet()
        tcep.init_net(1 + N*2193)
        tcep.register_test_data(error_xtest, error_ytest)
        tcep.fit(error_xtrain, error_ytrain,
                epochs = 500,
                validate_every = 10,
                model_name=self.model_name,
                input_name=f'{xtrain.shape[0]} datas',
                path = self.path
                )

        errors = np.zeros((101, 2193))

        errors[:N] = error_terms[:N]
        errors[N:] = tcep.predict(new_x)

        # Save the first few known results and stuff
        self._model = (linear_model, tcep, errors, normalization_term)

    def predict(self, xtest):
        # Retrieve models
        linear_model, tcep, errors, normalization_term = self.model
        ypred = np.zeros((xtest.shape[0], 2193))
        for i in range(xtest.shape[0]):
            # Calculate the indices
            # We need to extract a single feature but keep both dimensions, hence i:i+1
            x = xtest[i:i+1]
            error = errors[int(x/0.0075)]
            error = error * normalization_term
            linear_part = linear_model.predict(x)
            ypred[i] = linear_part + error

        return ypred

    @property
    def model_name(self):
        return "TCEP Model"

# Import antics
__all__ = [
    "Regressor",
    "RegressorFitError",
    "DecisionTreeRegression",
    "RidgeCVRegression",
    "GaussianRegression",
    "SGDRegression",
    "PassiveAggressiveRegression",
    "LinearRegression",
    "MultiTaskLassoCVRegression",
    "MultiTaskElasticNetCVRegression",
    "BayesianRidgeRegression",
    "SimpleNetRegression",
    "GLH1Regression",
    "GLH2Regression",
    "GLH3Regression",
    "GLH4Regression",
    "TCEPRegression"
]
