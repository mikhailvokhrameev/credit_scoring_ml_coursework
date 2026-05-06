import catboost as cb
import pandas as pd
import numpy as np
import mlflow
import re
from typing import Dict, Any, Tuple, Optional
from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost implementation optimized for native categorical feature handling.
    Employs 'Balanced' auto class weights for imbalanced datasets.
    """
    def _sanitize_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Removes special characters from column names"""
        X = X.copy()
        X.columns = [
            re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
            for c in X.columns
        ]
        return X


    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensures inference data has the exact same columns as training data"""
        X = X.copy()

        if not hasattr(self, "features_"):
            raise ValueError("Model is not fitted yet (features_ not found)")

        missing = set(self.features_) - set(X.columns)
        for col in missing:
            X[col] = 0

        return X[self.features_]


    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set = None) -> 'CatBoostModel':
        # Sanitize features
        X = self._sanitize_columns(X)
        
        if eval_set is not None:
            eval_set = (
                self._sanitize_columns(eval_set[0]),
                eval_set[1]
            )

        self.features_ = list(X.columns)
        
        # Identify categorical columns dynamically to leverage CatBoost's native powers
        cat_features =[col for col in X.columns if X[col].dtype.name in ['object', 'category']]
        
        device = self.params.get('device', 'cpu').upper()
        self.params['task_type'] = device
        
        if device == 'GPU':
            self.params.setdefault('devices', '0')
        
        self.params.setdefault('auto_class_weights', 'Balanced')
        self.params.setdefault('verbose', False)
        self.params.setdefault('allow_writing_files', False)
        
        self.model = cb.CatBoostClassifier(**self.params)
        
        eval_data = (eval_set[0], eval_set[1]) if eval_set else None
        
        # Manual parameter logging as MLflow CatBoost autologging is sometimes unstable in CV
        mlflow.log_params(self.params)
        
        self.model.fit(X, y, eval_set=eval_data, cat_features=cat_features, early_stopping_rounds=100 if eval_set else None)
        return self


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Sanitize and align internally before predicting
        X = self._sanitize_columns(X)
        X = self._align_features(X)
        return self.model.predict_proba(X)[:, 1]


    def get_feature_importance(self) -> pd.DataFrame:
        if getattr(self, "model", None) is None:
            raise ValueError("Model is not trained yet")
            
        importance = self.model.get_feature_importance()
        return pd.DataFrame({'feature': self.features_, 'importance': importance}).sort_values(by='importance', ascending=False)


    def get_optuna_space(self, trial) -> Dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 200, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "eval_metric": "AUC",
            "random_seed": 42
        }