import pandas as pd

# Caminho para o arquivo Excel
file_path = "Vehicles_export_prices_scaled_train_eng.xlsx"

# Ler o arquivo Excel
df = pd.read_excel(file_path)

# 1. Quantidade total de linhas no dataset
total_rows = df.shape[0]

# 2. Excluir linhas onde a variável-alvo (LAID_UP_TIME) está vazia
target = "LAID_UP_TIME"
rows_before = df.shape[0]
df = df.dropna(subset=[target])
rows_after = df.shape[0]
rows_removed = rows_before - rows_after

# 3. Distribuição das amostras de acordo com a classe de saída
class_distribution = df[target].value_counts().sort_values(ascending=False)

# 4. Para cada coluna, quantos valores diferentes aparecem
unique_values_per_column = df.nunique().sort_values(ascending=False)

# 5. Para cada coluna, quantos valores vazios existem
missing_values_per_column = df.isnull().sum().sort_values(ascending=False)

# 6. Estatísticas descritivas para LAID_UP_TIME
laid_up_description = df[target].describe()

# Salvar a saída em um arquivo de texto
output_path = "feedback_v2.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"1. Total de linhas no dataset: {total_rows}\n\n")
    f.write(f"2. Linhas removidas com LAID_UP_TIME vazio: {rows_removed}\n")
    f.write(f"   Linhas restantes: {rows_after}\n\n")
    f.write("3. Distribuição das amostras de acordo com a classe de saída (LAID_UP_TIME):\n")
    f.write(class_distribution.to_string())
    f.write("\n\n4. Valores diferentes por coluna:\n")
    f.write(unique_values_per_column.to_string())
    f.write("\n\n5. Valores vazios por coluna:\n")
    f.write(missing_values_per_column.to_string())
    f.write("\n\n6. Estatísticas descritivas para LAID_UP_TIME:\n")
    f.write(laid_up_description.to_string())

print(f"Arquivo de feedback gerado: {output_path}")
