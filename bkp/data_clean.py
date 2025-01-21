import pandas as pd

# Caminho do arquivo Excel
file_path = "Vehicles_export_prices_scaled_train_eng.xlsx"

# Ler o dataset
df = pd.read_excel(file_path)

# 1. Excluir linhas onde a variável-alvo (LAID_UP_TIME) está vazia
target = "LAID_UP_TIME"
df = df.dropna(subset=[target])
print(f"Linhas restantes após remover {target} vazio: {df.shape[0]}")

# 2. Identificar colunas com muitos valores vazios para possível exclusão
threshold = 0.9  # Excluir colunas com mais de 90% de valores vazios
cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > threshold]
print(f"Colunas com mais de {threshold*100}% valores vazios: {cols_to_drop}")

# Remover as colunas identificadas
df = df.drop(columns=cols_to_drop)
print(f"Colunas restantes após remoção: {df.shape[1]}")

# 3. Excluir colunas que são identificadores únicos e irrelevantes para modelagem
irrelevant_columns = ['CHASSIS_NUMBER', 'COMMISSION_NUMBER', 'RPAKREP_VEHICLE_HKEY']  # Adicione outras se necessário
df = df.drop(columns=[col for col in irrelevant_columns if col in df.columns])
print(f"Colunas restantes após remover identificadores únicos: {df.shape[1]}")

# 4. Tratar valores vazios restantes
# Para colunas numéricas: preencher com a média
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Para colunas categóricas: preencher com o valor mais frequente
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("Tratamento de valores vazios concluído.")

# 5. Exportar a base limpa para um novo arquivo
output_path = "base_limpa.xlsx"
df.to_excel(output_path, index=False)
print(f"Base limpa salva em: {output_path}")
