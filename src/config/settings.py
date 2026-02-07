"""
Configurações e constantes do projeto.
Design System: cores, estilos e constantes globais.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Paleta de cores
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    # Cores principais
    "primary": "#6366F1",       # Indigo vibrante
    "primary_light": "#818CF8",
    "primary_dark": "#4F46E5",
    
    # Cores secundárias
    "secondary": "#EC4899",     # Rosa accent
    "accent": "#14B8A6",        # Teal para destaque
    
    # Cores de estado
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    
    # Cores neutras
    "bg_dark": "#0F172A",       # Slate 900
    "bg_card": "#1E293B",       # Slate 800
    "bg_hover": "#334155",      # Slate 700
    "border": "#475569",        # Slate 600
    "text_primary": "#F8FAFC",  # Slate 50
    "text_secondary": "#94A3B8", # Slate 400
    "text_muted": "#64748B",    # Slate 500
    
    # Cores para modelos
    "logreg": "#6366F1",        # Indigo
    "rf": "#14B8A6",            # Teal
}

# Paleta para gráficos
MODEL_COLORS = {
    "logreg": COLORS["logreg"],
    "rf": COLORS["rf"],
}

MODEL_NAMES = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest"
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

TARGET_COLUMN = "income"
SENSITIVE_COLUMNS = ["sex", "race"]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DOS MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    "logreg": {
        "max_iter": 2000,
        "random_state": 42
    },
    "rf": {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1
    }
}

# Default threshold
DEFAULT_THRESHOLD = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DE CACHE
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_CONFIG = {
    "enabled": True,           # Ativar/desativar cache
    "cache_dir": ".cache",     # Diretório para ficheiros de cache
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DA APP
# ═══════════════════════════════════════════════════════════════════════════════

APP_CONFIG = {
    "title": "Model Evaluation Dashboard",
    "subtitle": "A Comparative Study of Classification Models Beyond Overall Accuracy",
    "host": "127.0.0.1",
    "port": 8050,
    "debug": True
}
