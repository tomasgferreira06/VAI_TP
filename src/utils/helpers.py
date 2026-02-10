from typing import Tuple
import pandas as pd


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Converte cor hex para rgba.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def filter_by_demographic(df: pd.DataFrame, analysis_focus: str) -> pd.DataFrame:
    """
    Filtra DataFrame por grupo demográfico.
    """
    # Para análise global, retorna todos os dados
    if analysis_focus == "global" or analysis_focus is None:
        return df
    
    # Para sex ou race, retornamos os dados
    return df


def get_demographic_groups(df: pd.DataFrame, analysis_focus: str) -> list:
    """
    Retorna lista de grupos demográficos para análise.
    """
    if analysis_focus == "global" or analysis_focus is None:
        return [("Global", df)]
    elif analysis_focus == "sex":
        return [
            ("Male", df[df["sex"] == "Male"]),
            ("Female", df[df["sex"] == "Female"])
        ]
    elif analysis_focus == "race":
        return [
            ("White", df[df["race"] == "White"]),
            ("Non-White", df[df["race"] != "White"])
        ]
    return [("Global", df)]
