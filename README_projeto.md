# Model Evaluation Dashboard

Dashboard interativo para avaliação e comparação de modelos de classificação binária, desenvolvido com Dash + Plotly.

## Estrutura do Projeto

```
VAI_TP/
├── run.py                  # Ponto de entrada principal
├── train.csv               # Dados de treino
├── test.csv                # Dados de teste
├── requirements.txt        # Dependências
├── README.md               # Este ficheiro
└── src/
    ├── __init__.py
    ├── app.py              # Configuração da app Dash
    │
    ├── config/             # Configurações e constantes
    │   ├── __init__.py
    │   ├── settings.py     # COLORS, MODEL_COLORS, MODEL_NAMES, configs
    │   └── styles.py       # CSS customizado
    │
    ├── data/               # Carregamento e processamento de dados
    │   ├── __init__.py
    │   └── loader.py       # load_data(), prepare_features()
    │
    ├── models/             # Treino e avaliação de modelos
    │   ├── __init__.py
    │   └── training.py     # train_pipelines(), métricas
    │
    ├── components/         # Componentes UI reutilizáveis
    │   ├── __init__.py
    │   ├── cards.py        # Metric cards, badges
    │   └── layout.py       # Header, sidebar
    │
    ├── charts/             # Funções de gráficos
    │   ├── __init__.py
    │   ├── view1_global.py     # Comparação global
    │   ├── view2_tradeoffs.py  # Trade-offs
    │   ├── view3_errors.py     # Análise de erros
    │   ├── view4_fairness.py   # Fairness
    │   └── advanced.py         # Parallel Coords, Radar, Sunburst
    │
    ├── layouts/            # Layouts das tabs
    │   ├── __init__.py
    │   └── tabs.py         # create_tab_*()
    │
    ├── callbacks/          # Callbacks Dash
    │   ├── __init__.py
    │   └── callbacks.py    # register_callbacks()
    │
    └── utils/              # Funções utilitárias
        ├── __init__.py
        └── helpers.py      # hex_to_rgba()
```

## Instalação

1. Criar ambiente virtual (recomendado):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Instalar dependências:
```bash
pip install -r requirements.txt
```

## Execução

```bash
python run.py
```

A aplicação estará disponível em: **http://127.0.0.1:8050**

## Features

### 5 Views Interativas

1. **Comparação Global**
   - Métricas principais (Accuracy, Precision, Recall, F1)
   - Curvas ROC com AUC
   - Feature Importance (Random Forest)
   - Calibration Plot com Brier Score

2. **Trade-offs**
   - Curvas Precision-Recall
   - Análise de métricas por threshold
   - Evolução FP/FN

3. **Análise de Erros**
   - Matriz de confusão interativa
   - Distribuição de erros por atributo
   - Comparação de taxas de erro

4. **Fairness**
   - Accuracy por grupo demográfico
   - Disparidade (gap) entre grupos
   - FPR/FNR por grupo

5. **Visualizações Avançadas**
   - Parallel Coordinates
   - Radar Chart
   - Sunburst Chart

### Funcionalidades Interativas

- **Threshold Slider**: Ajusta o limiar de decisão em tempo real
- **Model Selector**: Foco em modelo específico
- **Fairness Selector**: Escolha do atributo sensível (Sex/Race)
- **Reset Button**: Restaura valores padrão
- **Export Button**: Exporta métricas para CSV
- **Linked Brushing**: Seleção sincronizada entre gráficos

## Tecnologias

- **Dash** + **dash-bootstrap-components**: Framework web
- **Plotly**: Visualizações interativas
- **scikit-learn**: Modelos ML e métricas
- **pandas** + **numpy**: Manipulação de dados

## Dataset

Adult Income Dataset (Census Income):
- **Tarefa**: Classificação binária (rendimento >50K vs ≤50K)
- **Features**: Idade, educação, ocupação, etc.
- **Atributos sensíveis**: Sexo, Raça

## Modelos

1. **Logistic Regression** - Modelo baseline interpretável
2. **Random Forest** - Ensemble com feature importance

## Autor

Desenvolvido para a disciplina de **VAI** (Visualização e Análise de Informação)
