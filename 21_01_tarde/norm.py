import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from joblib import dump, load
import numpy as np

# Caminhos dos arquivos
file_train = "Data/Vehicles_export_prices_scaled_train_eng.xlsx"
file_test = "Data/Vehicles_export_prices_scaled_stud_test_eng.xlsx"

# Ler as bases
df_train = pd.read_excel(file_train)
df_test = pd.read_excel(file_test)

# Concatenar as bases
df_combined = pd.concat([df_train, df_test], ignore_index=True)
df_combined = df_combined.drop(columns=["PAINT_TYPE"])

# Remover ou converter colunas datetime64
datetime_cols = df_combined.select_dtypes(include=["datetime64"]).columns.tolist()
for col in datetime_cols:
    df_combined[col] = (df_combined[col] - pd.Timestamp("1970-01-01")).dt.total_seconds() / (60 * 60 * 24)

# Identificar colunas categóricas e numéricas
cat_cols = df_combined.select_dtypes(include=["object"]).columns.tolist()
num_cols = df_combined.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Codificar variáveis categóricas com LabelEncoder
label_encoder = LabelEncoder()
for col in cat_cols:
    df_combined[col] = label_encoder.fit_transform(df_combined[col].astype(str))

# Escalar todas as colunas numéricas (exceto o target "LAID_UP_TIME")
scaler = StandardScaler()
cols_to_scale = [col for col in num_cols + cat_cols if col != "LAID_UP_TIME"]
df_combined[cols_to_scale] = scaler.fit_transform(df_combined[cols_to_scale])

# Separar novamente os conjuntos
df_train_encoded = df_combined.iloc[:len(df_train)]
df_test_encoded = df_combined.iloc[len(df_train):]

# Limpar linhas com `LAID_UP_TIME` vazio no conjunto de treino
target = "LAID_UP_TIME"
df_train_encoded = df_train_encoded.dropna(subset=[target])

# Separar variáveis explicativas e alvo
X = df_train_encoded.drop(columns=[target])
y = df_train_encoded[target]
X_test = df_test_encoded.drop(columns=[target], errors="ignore")  # Teste pode não ter target

# Verificar e tratar valores ausentes
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Divisão 70-15-15
X_train_full, X_temp, y_train_full, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test_final, y_val, y_test_final = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Validação cruzada com 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
for train_idx, val_idx in kf.split(X_train_full):
    X_train, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    # Treinar MLP
    mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mlp_model.fit(X_train, y_train)

    # Previsão no fold de validação
    y_val_pred_fold = mlp_model.predict(X_val_fold)
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred_fold))
    rmse_scores.append(fold_rmse)

print(f"RMSE médio nos folds de validação: {np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})")

# Treinar no conjunto completo de treino após validação cruzada
mlp_model.fit(X_train_full, y_train_full)

# Avaliação no conjunto de validação final
y_val_pred = mlp_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE no conjunto de validação final: {val_rmse:.2f}")

# Fazer previsões no conjunto de teste
y_test_pred = mlp_model.predict(X_test_final)
final_rmse = np.sqrt(mean_squared_error(y_test_final, y_test_pred))
print(f"RMSE no conjunto de teste final: {final_rmse:.2f}")

# Exibir algumas previsões e seus erros no conjunto de teste
result_test = pd.DataFrame({
    "CHASSIS": df_test_encoded.get("CHASSIS", [None] * len(y_test_pred)),
    "Real": y_test_final,
    "Previsto": y_test_pred,
    "Erro": y_test_final - y_test_pred
})
print(result_test.head())

# Salvar as previsões em um arquivo Excel
result_test.to_excel("result_test_predictions_norm.xlsx", index=False)

# Salvar o modelo treinado para uso futuro
dump(mlp_model, "mlp_model_norm.joblib")
