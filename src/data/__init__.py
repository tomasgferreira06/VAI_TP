"""
Módulo de dados e cache.
"""
from src.data.loader import load_data, prepare_features, get_column_types
from src.data.cache import cache_exists, load_cache, save_cache, clear_cache, get_cache_info

__all__ = [
    "load_data", 
    "prepare_features", 
    "get_column_types",
    "cache_exists",
    "load_cache",
    "save_cache",
    "clear_cache",
    "get_cache_info"
]
