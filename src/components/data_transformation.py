import sys
from dataclasses import dataclass
from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation parameters."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        # Define columns as class attributes to avoid repetition
        self.numerical_columns = ['writing score', 'reading score']
        self.categorical_columns = [
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
        ]
        self.target_column = 'math score'

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a preprocessing pipeline for numerical and categorical features.
        
        Returns:
            ColumnTransformer: Combined preprocessing pipeline for all features
        
        Raises:
            CustomException: If there's an error in creating the transformer
        """
        try:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {self.categorical_columns}")
            logging.info(f"Numerical columns: {self.numerical_columns}")

            return ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, self.numerical_columns),
                    ('cat_pipeline', cat_pipeline, self.categorical_columns)
                ]
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, 
        train_path: str, 
        test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Reads train and test datasets, applies preprocessing, and returns transformed arrays.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            
        Returns:
            tuple: (transformed training array, transformed test array, preprocessor path)
            
        Raises:
            CustomException: If there's an error in the transformation process
        """
        try:
            # Load and validate data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            required_columns = self.numerical_columns + self.categorical_columns + [self.target_column]
            for df, name in [(train_df, 'train'), (test_df, 'test')]:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Missing columns in {name} data: {missing_cols}")

            logging.info("Successfully validated data schemas")
            
            # Get preprocessing object and prepare data
            preprocessor = self.get_data_transformer_object()
            
            def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
                features = df.drop(columns=[self.target_column])
                target = df[self.target_column]
                return features, target
            
            # Split and transform data
            X_train, y_train = prepare_features(train_df)
            X_test, y_test = prepare_features(test_df)
            
            logging.info("Applying preprocessing transformations")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combine features and target
            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            # Save preprocessor
            preprocessor_path = self.config.preprocessor_obj_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            save_object(file_path=preprocessor_path, obj=preprocessor)
            
            logging.info("Data transformation completed successfully")
            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)



    