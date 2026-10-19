"""
preprocessing.py – Data loading, cleaning, feature engineering, and pipeline construction.
Designed for complete leakage prevention: pipelines fit strictly on training folds.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(filepath: str) -> pd.DataFrame:
    """Loads the original Telco Customer Churn CSV."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataset:
    1. Converts TotalCharges to numeric (blank spaces -> NaN).
    2. Drops rows with missing TotalCharges (11 rows where tenure=0, < 0.2% of data).
    3. Drops customerID identifier column.
    
    Decision justification:
    - TotalCharges is an object due to 11 empty space strings (' ') for brand-new customers (tenure=0).
    - Dropping 11 rows out of 7,043 (<0.16%) does not bias the distribution or discard meaningful history.
    - customerID is a high-cardinality unique identifier providing zero generalizable signal.
    """
    df_clean = df.copy()
    
    # 1. TotalCharges type conversion
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # 2. Drop 11 missing TotalCharges rows
    df_clean = df_clean.dropna(subset=['TotalCharges']).reset_index(drop=True)
    
    # 3. Drop customerID
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        
    return df_clean


def engineer_features(df: pd.DataFrame, include_service_count: bool = True, include_tenure_band: bool = False) -> pd.DataFrame:
    """
    Engineers domain-relevant features:
    
    1. SubscribedServiceCount:
       Correctly counts active subscriptions across 9 telecom services:
       - PhoneService == 'Yes'
       - MultipleLines == 'Yes'
       - InternetService in ['DSL', 'Fiber optic']
       - OnlineSecurity == 'Yes'
       - OnlineBackup == 'Yes'
       - DeviceProtection == 'Yes'
       - TechSupport == 'Yes'
       - StreamingTV == 'Yes'
       - StreamingMovies == 'Yes'
       Note: InternetService values are 'DSL', 'Fiber optic', or 'No' (never 'Yes').
       
    2. TenureBand (Optional, evaluated via CV):
       Categorizes continuous tenure into 6 yearly bands.
       If CV testing shows it adds no value or increases redundancy, it is excluded.
    """
    df_eng = df.copy()
    
    if include_service_count:
        service_flags = [
            (df_eng['PhoneService'] == 'Yes'),
            (df_eng['MultipleLines'] == 'Yes'),
            (df_eng['InternetService'].isin(['DSL', 'Fiber optic'])),
            (df_eng['OnlineSecurity'] == 'Yes'),
            (df_eng['OnlineBackup'] == 'Yes'),
            (df_eng['DeviceProtection'] == 'Yes'),
            (df_eng['TechSupport'] == 'Yes'),
            (df_eng['StreamingTV'] == 'Yes'),
            (df_eng['StreamingMovies'] == 'Yes')
        ]
        # Sum boolean flags along axis 0
        df_eng['SubscribedServiceCount'] = sum(flag.astype(int) for flag in service_flags)
        
    if include_tenure_band:
        df_eng['TenureBand'] = pd.cut(
            df_eng['tenure'],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=['0-1 Yr', '1-2 Yrs', '2-3 Yrs', '3-4 Yrs', '4-5 Yrs', '5-6 Yrs'],
            include_lowest=True
        ).astype(str)
        
    return df_eng


def get_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for preprocessing:
    - Numeric features: Median imputation (robust to potential skew) + StandardScaler.
    - Categorical features: Most-frequent imputation + OneHotEncoder(drop='first', handle_unknown='ignore').
      drop='first' prevents multicollinearity (dummy variable trap) for linear models.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor


def prepare_data(df: pd.DataFrame, target_col: str = 'Churn'):
    """
    Splits features and target using stratified 80/20 train/test split.
    Identifies column types and instantiates the preprocessor.
    
    CRITICAL: Does NOT fit the preprocessor here.
    Returns raw X_train, X_test, y_train, y_test to guarantee zero CV leakage.
    """
    X = df.drop(columns=[target_col])
    y = (df[target_col] == 'Yes').astype(int)
    
    # Stratified split preserves exact class distribution (26.58% churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    numerical_cols = X_train.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = get_preprocessor(numerical_cols, categorical_cols)
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols
