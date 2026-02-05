"""
Módulo de cache para evitar re-treino de modelos.

Guarda o DataFrame de avaliação e os pipelines treinados em disco,
permitindo carregamento rápido em hot-reloads.
"""
import os
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import CACHE_CONFIG


def _get_cache_dir() -> Path:
    """Retorna o diretório de cache, criando-o se necessário."""
    cache_dir = Path(CACHE_CONFIG["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _compute_data_hash(train_path: str, test_path: str) -> str:
    """
    Calcula hash dos ficheiros de dados para detetar alterações.
    
    Args:
        train_path: Caminho para ficheiro de treino
        test_path: Caminho para ficheiro de teste
        
    Returns:
        Hash MD5 combinado dos ficheiros
    """
    hasher = hashlib.md5()
    
    for filepath in [train_path, test_path]:
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                # Ler em chunks para ficheiros grandes
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
    
    return hasher.hexdigest()[:12]


def _get_cache_paths(data_hash: str) -> Tuple[Path, Path]:
    """Retorna os caminhos dos ficheiros de cache."""
    cache_dir = _get_cache_dir()
    eval_path = cache_dir / f"eval_df_{data_hash}.pkl"
    pipelines_path = cache_dir / f"pipelines_{data_hash}.pkl"
    return eval_path, pipelines_path


def cache_exists(train_path: str = "train.csv", test_path: str = "test.csv") -> bool:
    """
    Verifica se existe cache válido para os dados atuais.
    
    Args:
        train_path: Caminho para ficheiro de treino
        test_path: Caminho para ficheiro de teste
        
    Returns:
        True se cache existe e é válido
    """
    if not CACHE_CONFIG["enabled"]:
        return False
    
    data_hash = _compute_data_hash(train_path, test_path)
    eval_path, pipelines_path = _get_cache_paths(data_hash)
    
    return eval_path.exists() and pipelines_path.exists()


def save_cache(
    eval_df: pd.DataFrame,
    pipelines: Dict[str, Pipeline],
    train_path: str = "train.csv",
    test_path: str = "test.csv"
) -> None:
    """
    Guarda o DataFrame de avaliação e pipelines em cache.
    
    Args:
        eval_df: DataFrame com predições e métricas
        pipelines: Dicionário de pipelines treinados
        train_path: Caminho para ficheiro de treino
        test_path: Caminho para ficheiro de teste
    """
    if not CACHE_CONFIG["enabled"]:
        return
    
    data_hash = _compute_data_hash(train_path, test_path)
    eval_path, pipelines_path = _get_cache_paths(data_hash)
    
    # Guardar DataFrame de avaliação
    with open(eval_path, "wb") as f:
        pickle.dump(eval_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Guardar pipelines
    with open(pipelines_path, "wb") as f:
        pickle.dump(pipelines, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"[CACHE] Dados guardados em: {_get_cache_dir()}")


def load_cache(
    train_path: str = "train.csv",
    test_path: str = "test.csv"
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Carrega o DataFrame de avaliação e pipelines do cache.
    
    Args:
        train_path: Caminho para ficheiro de treino
        test_path: Caminho para ficheiro de teste
        
    Returns:
        Tuple com (eval_df, pipelines)
        
    Raises:
        FileNotFoundError: Se cache não existe
    """
    data_hash = _compute_data_hash(train_path, test_path)
    eval_path, pipelines_path = _get_cache_paths(data_hash)
    
    if not eval_path.exists() or not pipelines_path.exists():
        raise FileNotFoundError("Cache não encontrado")
    
    with open(eval_path, "rb") as f:
        eval_df = pickle.load(f)
    
    with open(pipelines_path, "rb") as f:
        pipelines = pickle.load(f)
    
    print(f"[CACHE] Dados carregados do cache (hash: {data_hash})")
    return eval_df, pipelines


def clear_cache() -> None:
    """Remove todos os ficheiros de cache."""
    cache_dir = _get_cache_dir()
    
    removed = 0
    for file in cache_dir.glob("*.pkl"):
        file.unlink()
        removed += 1
    
    print(f"[CACHE] {removed} ficheiro(s) removido(s)")


def get_cache_info() -> Dict:
    """
    Retorna informação sobre o estado do cache.
    
    Returns:
        Dicionário com informação do cache
    """
    cache_dir = _get_cache_dir()
    
    files = list(cache_dir.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in files)
    
    return {
        "enabled": CACHE_CONFIG["enabled"],
        "directory": str(cache_dir),
        "files_count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }
