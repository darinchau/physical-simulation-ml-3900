from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression as Linear, RidgeCV, MultiTaskElasticNetCV, MultiTaskLassoCV, ElasticNetCV, LassoCV, BayesianRidge, OrthogonalMatchingPursuitCV, SGDRegressor as SGD, PassiveAggressiveRegressor as PassiveAggressive
from sklearn.tree import DecisionTreeRegressor as DecisionTree
import numpy as np
from abc import abstractmethod as virtual
from load import index_exclude, index_include
import warnings

# Filter all scikitlearn warnings
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

    @property
    def trained(self):
        return hasattr(self, "_model")

    def predict(self, xtest):
        return self.model.predict(xtest)

    @property
    def train_info(self):
        st = f"Trained on {self.model_name} with:"
        for k, v in self.__dict__.items():
            if k == "_model":
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

        xtrain = xtrain.reshape(-1, 1)
        xtest = xtest.reshape(-1, 1)

        return xtrain, xtest, ytrain, ytest

    def calculate_error(self, model, xtest, ytest, skip_error, verbose):
        # Calculate the error
        if skip_error:
            return model, None

        # Calculate error
        ypred = self.predict(xtest)

        # Calculate the mean square error
        mse = np.mean((ytest - ypred)**2)
        worst = np.max(np.abs(ytest - ypred))

        if verbose:
            print(f"{self.model_name} - MSE: {mse}, worse: {worst}")

        return model, (mse, worst)

    # Preprocess, fit data, and calculate error
    def fit(self, inputs, raw_data, training_idx, verbose = False, skip_error = False):
        # Split data
        xtrain, xtest, ytrain, ytest = self.preprocess(inputs, raw_data, training_idx)

        # Train the model
        np.random.seed(12345)
        model = self.fit_model(xtrain, ytrain)

        # Calculate and return error
        model, err = self.calculate_error(model, xtest, ytest, skip_error, verbose)

        self._model = model

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
    def fit_model(self, xtrain, ytrain, verbose=False, skip_error=False):
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

        err = self.calculate_error(xtest, ytest, skip_error, verbose)

        self._model = models

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

#### Neural networks ####
