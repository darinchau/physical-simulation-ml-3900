## This module contains all models. Hopefully less of a clusterfcusk than the last one
from __future__ import annotations
from abc import ABC, abstractmethod as virtual
from torch import Tensor

class TrainingError:
    """Indicates error during training that can be safely skipped"""
    pass

class Model:
    """Interface for all model-like structures"""
    @virtual
    def fit_logic(self, xtrain: Tensor, ytrain: Tensor) -> Model:
        """The train logic of the model. Every model inherited from Model needs to implement this
        This needs to return the model, which will be passed in the predict_logic method as an argument"""
        raise NotImplementedError
    
    @virtual
    def predict_logic(self, model, xtest: Tensor) -> Tensor:
        """The prediction logic of the model. Every model inherited from Model needs to implement this.
        Takes in xtest and the model which will be exactly what the model has returned in fit_logic method"""
        raise NotImplementedError
    
    @virtual
    def save(self, root: str, name: str):
        """Save your models in the folder 'root', with name being the training input name"""
        raise NotImplementedError
    
    @virtual
    def load(self, path: str):
        """Loads the model on a self object, overwriting any previous data"""
        raise NotImplementedError
    
    @property
    def trained(self) -> bool:
        if not hasattr(self, "_trained"):
            return False
        return self._trained
    
    @property
    def informed(self) -> dict[str, Tensor]:
        if hasattr(self, "_testing") and self._testing:
            raise ValueError("Model tried to access informed data during testing phase which is not permitted")
        if not hasattr(self, "_informed"):
            self._informed = {}
        return self._informed
    
    def inform(self, info: Tensor, name: str):
        """Informs the model about a certain information. Whether the model uses it depends on the implementation
        This informed stuff will only be available during training and not during testing"""
        self.informed[name] = info 
    
    def fit(self, xtrain: Tensor, ytrain: Tensor):
        """Fit xtrain and ytrain on the model"""
        try:
            self._model = self.fit_logic(xtrain, ytrain)
        except ValueError as e:
            if f"{e}".startswith("Input X contains infinity"):
                raise TrainingError("Caught exploding coefficients during training")
            raise e

        self._ytrain_shape = ytrain.shape
        self._xtrain_shape = xtrain.shape
        self._trained = True
        return self
    
    # Predict inner logic + sanity checks
    def predict(self, xtest: Tensor) -> Tensor:
        """Use xtest to predict ytest"""        
        if not self.trained:
            raise ValueError("Model has not been trained")

        if xtest.shape[1:] != self._xtrain_shape[1:]:
            a = ("_",) + self._xtrain_shape[1:]
            raise ValueError(f"Expects xtest to have shape ({a}) from model training, but got xtest with shape {xtest.shape}")
        
        # Prediction
        self._testing = True
        try:
            ypred = self.predict_logic(self._model, xtest)
        except ValueError as e:
            if f"{e}".startswith("Input X contains infinity"):
                raise TrainingError("Caught exploding coefficients during training")
            raise e
        self._testing = False

        # Sanity checks
        if ypred.shape[0] != len(xtest):
            raise ValueError(f"There are different number of samples in xtest ({len(xtest)}) and ypred ({ypred.shape[0]})")
        
        # Wrap back ypred in the correct shape
        if ypred.shape[1:] != self._ytrain_shape[1:]:
            raise ValueError("The shape of ypred is different than ytrain")

        return ypred
