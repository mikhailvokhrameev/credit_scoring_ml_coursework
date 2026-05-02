import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class ApplicationPreprocessor:
    def __init__(self):
        self.label_encoders = {}

    def fix_anomalies(self, df: pd.DataFrame):
        df = df.copy()

        if 'DAYS_EMPLOYED' in df.columns:
            df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(np.int8) # Create anomaly flag
            df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
            df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs() # Domain correction
            
        if 'DAYS_BIRTH' in df.columns:
            df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs() # Domain correction
            
        df.replace({'XNA': np.nan, 'Unknown': np.nan}, inplace=True) # Replace artifacts with NaN
        return df

    def fit_transform_categorical(self, train: pd.DataFrame, test: pd.DataFrame):
        """Common LabelEncoding for train and test"""
        
        train = train.copy()
        test = test.copy()
        
        cat_cols = [col for col in train.columns if train[col].dtype == 'object']
        
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values)) # Safe encoding to avoid unseen labels
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
            self.label_encoders[col] = le
            
        logger.info("Categorical features encoded | num_cols=%s", len(cat_cols))
        return train, test