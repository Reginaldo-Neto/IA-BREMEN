import pandas as pd

# Caminho para o arquivo Excel
file_path = "Vehicles_export_prices_scaled_train_eng.xlsx"

# Ler o dataset
df = pd.read_excel(file_path)

# Função para analisar cada característica
def analyze_column(df, col, target):
    analysis = f"Coluna: {col}\n"
    analysis += f"Tipo: {df[col].dtype}\n"
    analysis += f"Valores únicos: {df[col].nunique()}\n"
    analysis += f"Valores vazios: {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.2f}%)\n"
    
    if df[col].dtype in ['float64', 'int64']:
        # Preencher valores nulos para evitar erros
        temp_col = df[col].fillna(0)
        
        # Estatísticas descritivas para numéricas
        analysis += f"Estatísticas:\n{temp_col.describe()}\n"
        
        # Verificar se a variância não é zero antes de calcular a correlação
        if temp_col.std() > 0 and target in df.columns and df[target].std() > 0:
            correlation = temp_col.corr(df[target])
            analysis += f"Correlação com {target}: {correlation:.4f}\n"
        else:
            analysis += f"Correlação com {target}: Não aplicável (variância zero ou alvo inválido)\n"
    elif df[col].dtype == 'object':
        # Valores mais frequentes para categóricas
        analysis += "Valores mais frequentes:\n"
        analysis += df[col].value_counts().head(5).to_string() + "\n"
    
    analysis += "-" * 50 + "\n"
    return analysis

# Analisar cada coluna
target = "LAID_UP_TIME"
output_path = "feature_analysis.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for col in df.columns:
        f.write(analyze_column(df, col, target))

print(f"Análise de características salva em: {output_path}")
