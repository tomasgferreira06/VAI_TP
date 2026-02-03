"""
Módulo de treino e avaliação de modelos.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from src.config.settings import MODEL_CONFIG, SENSITIVE_COLUMNS


def create_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """
    Cria o preprocessador para as features.
    
    Args:
        cat_cols: Lista de colunas categóricas
        num_cols: Lista de colunas numéricas
        
    Returns:
        ColumnTransformer configurado
    """
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_cols),
            ("num", numeric_transformer, num_cols),
        ],
        remainder="drop"
    )
    
    return preprocessor


def create_models() -> Dict:
    """
    Cria os modelos de classificação.
    
    Returns:
        Dicionário com os modelos
    """
    return {
        "logreg": LogisticRegression(**MODEL_CONFIG["logreg"]),
        "rf": RandomForestClassifier(**MODEL_CONFIG["rf"])
    }


def train_pipelines(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    cat_cols: list,
    num_cols: list
) -> Dict[str, Pipeline]:
    """
    Treina os pipelines de modelos.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        cat_cols: Colunas categóricas
        num_cols: Colunas numéricas
        
    Returns:
        Dicionário com pipelines treinados
    """
    preprocessor = create_preprocessor(cat_cols, num_cols)
    models = create_models()
    
    pipelines = {}
    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
    
    print("[OK] Modelos treinados.")
    return pipelines


def predict_table(
    pipe: Pipeline, 
    X: pd.DataFrame, 
    y: pd.Series, 
    model_name: str, 
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Gera tabela de predições para um modelo.
    
    Args:
        pipe: Pipeline treinado
        X: Features
        y: Target real
        model_name: Nome do modelo
        threshold: Limiar de decisão
        
    Returns:
        DataFrame com predições
    """
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame({
        "sample_id": np.arange(len(y)),
        "model": model_name,
        "y_true": y.values,
        "y_proba": proba,
        "y_pred": pred
    })

    # Adicionar sensitive cols
    for c in SENSITIVE_COLUMNS:
        out[c] = X[c].values

    return out


def create_evaluation_df(
    pipelines: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Cria DataFrame de avaliação com todos os modelos.
    
    Args:
        pipelines: Dicionário de pipelines
        X_test: Features de teste
        y_test: Target de teste
        
    Returns:
        DataFrame consolidado
    """
    return pd.concat(
        [predict_table(pipelines[name], X_test, y_test, name) for name in pipelines.keys()],
        ignore_index=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

def global_metrics(df: pd.DataFrame) -> pd.Series:
    """Calcula métricas globais."""
    return pd.Series({
        "accuracy": accuracy_score(df["y_true"], df["y_pred"]),
        "precision": precision_score(df["y_true"], df["y_pred"], zero_division=0),
        "recall": recall_score(df["y_true"], df["y_pred"], zero_division=0),
        "f1": f1_score(df["y_true"], df["y_pred"], zero_division=0),
    })


def confusion_parts(df: pd.DataFrame) -> pd.Series:
    """Extrai partes da matriz de confusão."""
    tn, fp, fn, tp = confusion_matrix(df["y_true"], df["y_pred"]).ravel()
    return pd.Series({"tn": tn, "fp": fp, "fn": fn, "tp": tp})


def group_fairness_metrics(df: pd.DataFrame) -> pd.Series:
    """Calcula métricas de fairness por grupo."""
    tn, fp, fn, tp = confusion_matrix(df["y_true"], df["y_pred"]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return pd.Series({
        "accuracy": accuracy_score(df["y_true"], df["y_pred"]),
        "fpr": fpr,
        "fnr": fnr,
        "support": len(df)
    })


def recompute_with_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Recalcula predições com novo threshold."""
    df = df.copy()
    df["y_pred"] = (df["y_proba"] >= threshold).astype(int)
    return df
