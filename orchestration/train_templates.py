from models.knn import KNNModel
from models.lightgbm import LGBMModel
from models.svm import SVMModel
from models.tree import TreeModel
from models.xgboost import XGBoostModel
from sklearn.model_selection import train_test_split


class TrainModels:

    def __init__(self, templates):
        self.models_mapping = { "tree": TreeModel,
                                "knn": KNNModel,
                                "lightgbm":LGBMModel,
                                "svm": SVMModel,
                                "xgboost": XGBoostModel
                                }
        self.templates = templates
        return
    

    def _initialize_models(self, templates):
        model_objects = []
        for model_dict in templates:
            model_key = model_dict["model"]
            init_params = model_dict["init"]
            model_objects.append(self.models_mapping[model_key](**init_params))
        return model_objects
    

    def train_models(self, df, features, target, cv=5, test_size=0.2, random_state=42):
        templates = self.templates
        mdl_objects = self._initialize_models(templates)

        X = df[features]
        y = df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  
        for i_mdl, mdl_obj in enumerate(mdl_objects):
            mdl_obj.fit(self.X_train, self.y_train, **templates[i_mdl]["fit"])
        return mdl_objects
