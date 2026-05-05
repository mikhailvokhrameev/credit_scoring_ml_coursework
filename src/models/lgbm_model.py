import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.models.base import BaseModel
import re

class LGBMModel(BaseModel):
    """
    LightGBM implementation for credit default risk prediction.
    Features modern Early Stopping callback to avoid deprecated arguments.
    """
    def _sanitize_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Removes special characters from column names to comply with LightGBM constraints"""
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
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set = None) -> 'LGBMModel':
         # Sanitize features
        X = self._sanitize_columns(X)
        
        if eval_set is not None:
            eval_set = (
                self._sanitize_columns(eval_set[0]),
                eval_set[1]
            )

        self.features_ = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)

        callbacks = []
        eval_data = None

        if eval_set is not None:
            eval_data = [(eval_set[0], eval_set[1])]
            # Modern callback format for LightGBM
            callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))
        
        self.model.fit(X, y, eval_set=eval_data, eval_metric="auc", callbacks=callbacks)

        return self


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Sanitize and align internally before predicting
        X = self._sanitize_columns(X)
        X = self._align_features(X)
        return self.model.predict_proba(X)[:, 1]


    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        importance = self.model.booster_.feature_importance(importance_type="gain")

        return pd.DataFrame({"feature": self.features_, "importance": importance}).sort_values(by="importance", ascending=False)


    def get_optuna_space(self, trial) -> Dict[str, Any]:
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1
        }