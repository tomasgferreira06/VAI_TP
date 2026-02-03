import numpy as np
import pandas as pd

# Configurações
pd.set_option("display.max_columns", 200)
np.random.seed(42)


def main():
    """Função principal que inicializa e executa o dashboard."""
    print("=" * 70)
    print("MODEL EVALUATION DASHBOARD")
    print("=" * 70)
    
    # 1. Carregar dados
    print("\n[1/5] A carregar dados...")
    from src.data.loader import load_data, prepare_features, get_column_types
    
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)
    cat_cols, num_cols = get_column_types(X_train)
    
    print(f"      Train: {X_train.shape[0]:,} amostras")
    print(f"      Test:  {X_test.shape[0]:,} amostras")
    
    # 2. Treinar modelos
    print("\n[2/5] A treinar modelos...")
    from src.models.training import train_pipelines, create_evaluation_df
    
    pipelines = train_pipelines(X_train, y_train, cat_cols, num_cols)
    
    # 3. Criar DataFrame de avaliação
    print("\n[3/5] A criar dados de avaliacao...")
    eval_df = create_evaluation_df(pipelines, X_test, y_test)
    print(f"      DataFrame: {len(eval_df):,} linhas")
    
    # 4. Criar aplicação
    print("\n[4/5] A inicializar aplicacao Dash...")
    from src.app import create_app
    from src.callbacks import register_callbacks
    
    app = create_app(
        test_samples=len(X_test),
        positive_rate=y_test.mean()
    )
    
    # 5. Registar callbacks
    print("\n[5/5] A registar callbacks...")
    register_callbacks(app, eval_df, pipelines, cat_cols, num_cols)
    
    print("\n" + "=" * 70)
    print("DASHBOARD PRONTO!")
    print("=" * 70)
    print("\nAcesse: http://127.0.0.1:8050")
    print("Para parar: Ctrl+C")
    print()
    
    # Executar
    app.run(debug=True, port=8050, dev_tools_hot_reload=True)


if __name__ == "__main__":
    main()
