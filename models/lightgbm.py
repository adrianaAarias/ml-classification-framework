from models.template import Model
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


class LGBMModel(Model):
    def __init__(self, grid_params=None, scale_input=False):
        if grid_params is None:
            self.grid_params = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.1, 0.01, 0.001],
                'n_estimators': [100, 300, 500]
            }
        else:
            self.grid_params = grid_params
        
        self.scale_input = scale_input

    def _search_best_params(self, X, y, cv, score_metric):
        mdl = GridSearchCV(LGBMClassifier(), self.grid_params, cv=cv, scoring=score_metric)
        mdl.fit(X, y)
        return mdl.best_params_


    def fit(self, X, y, feature_list=[], cv=5, score_metric='f1'):
        if len(feature_list) == 0:
            feature_list = X.columns
        self.feature_list = feature_list

        if self.scale_input:
            self.scaler = MinMaxScaler()
            X_train = self.scaler.fit_transform(X[feature_list])
        else:
            X_train = X[feature_list]
        
        params = self._search_best_params(X_train, y, cv, score_metric)
        mdl = LGBMClassifier(**params)
        mdl.fit(X_train, y)
        self.mdl = mdl
        return mdl


    def transform(self, X):
        if self.scale_input:
            X_pred = self.scaler.transform(X[self.feature_list])
        else:
            X_pred = X[self.feature_list]
        y_pred = self.mdl.predict_proba(X_pred)
        return y_pred