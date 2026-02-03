"""
Módulo de carregamento e processamento de dados.
"""
import numpy as np
import pandas as pd
from typing import Tuple

from src.config.settings import ADULT_COLUMNS, TARGET_COLUMN, SENSITIVE_COLUMNS


def standardize_adult(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza o dataset Adult Income.
    
    Args:
        df: DataFrame com dados brutos
        
    Returns:
        DataFrame padronizado
    """
    df = df.copy()
    df.columns = ADULT_COLUMNS

    # Strip em strings + trocar '?' por NaN
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace("?", np.nan)

    # Limpar a label (no test vem <=50K. e >50K.)
    df["income"] = df["income"].str.replace(".", "", regex=False)

    # Mapear label para 0/1
    df["income"] = df["income"].map({">50K": 1, "<=50K": 0}).astype(int)

    return df


def load_data(train_path: str = "train.csv", test_path: str = "test.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os datasets de treino e teste.
    
    Args:
        train_path: Caminho para o ficheiro de treino
        test_path: Caminho para o ficheiro de teste
        
    Returns:
        Tuple com (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = standardize_adult(train_df)
    test_df = standardize_adult(test_df)
    
    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separa features e target.
    
    Args:
        train_df: DataFrame de treino
        test_df: DataFrame de teste
        
    Returns:
        Tuple com (X_train, y_train, X_test, y_test)
    """
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].copy()
    
    return X_train, y_train, X_test, y_test


def get_column_types(X: pd.DataFrame) -> Tuple[list, list]:
    """
    Identifica colunas categóricas e numéricas.
    
    Args:
        X: DataFrame de features
        
    Returns:
        Tuple com (cat_cols, num_cols)
    """
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols
