"""
Funções utilitárias.
"""
from typing import Tuple


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Converte cor hex para rgba.
    
    Args:
        hex_color: Cor em formato hexadecimal (#RRGGBB)
        alpha: Valor de transparência (0.0 a 1.0)
        
    Returns:
        String rgba no formato "rgba(r,g,b,a)"
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
