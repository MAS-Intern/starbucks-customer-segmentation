"""
Data preprocessing module for Starbucks Customer Segmentation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
    
    def clean_data(self, df):
        """Clean the dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Remove outliers using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def preprocess(self, df, feature_cols):
        """Scale and prepare features for modeling"""
        df_clean = self.clean_data(df)
        X = df_clean[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, df_clean
    
    def inverse_transform(self, X_scaled):
        """Inverse transform scaled data for interpretation"""
        return self.scaler.inverse_transform(X_scaled)
