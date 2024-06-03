from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def _search_best_params(self,X,y,cv, score_metric):
        """Search best hyperparams"""
        pass


    @abstractmethod
    def fit(self, X, y, feature_list = [], cv=5, score_metric='f1'):
        """Training algorithm for the machine learning model"""
        pass


    @abstractmethod
    def transform(self, x):
        """Prediction of new data"""
        pass