from models.template import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class KNNModel(Model):
    def __init__(self, grid_params = None, scale_input = False):
        if grid_params is None:
            self.grid_params = {"n_neighbors":list(range(1,31)), "weights":["uniform", "distance"]}
        else:
            self.grid_params = grid_params
        
        self.scale_input = scale_input
        return
    

    def _search_best_params(self,X,y,cv, score_metric):
        mdl = GridSearchCV(KNeighborsClassifier(), self.grid_params, cv=cv,scoring=score_metric) 
        mdl.fit(X, y)
        return mdl.best_params_
        

    def fit(self, X, y, feature_list = [], cv=5, score_metric='f1'):
        if len(feature_list) == 0:
            feature_list = X.columns
        self.feature_list = feature_list

        if self.scale_input:
            self.scaler = MinMaxScaler()
            X_train = self.scaler.fit_transform(X[feature_list])
        else:
            X_train = X[feature_list]
        
        
        params = self._search_best_params(X_train,y,cv, score_metric)
        mdl = KNeighborsClassifier(**params)
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
        