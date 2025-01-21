import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor  # Exemplo de modelo
from sklearn.metrics import mean_squared_error
import numpy as np

# Carregar o Dataset 1 (limpo)
dataset1_path = "cleaned_train_dataset_v2.xlsx"
df = pd.read_excel(dataset1_path)

# Excluir linhas com valores vazios na coluna 'LAID_UP_TIME'
df = df.dropna(subset=['LAID_UP_TIME'])

# Separar features (X) e target (y)
X = df.drop(columns=['LAID_UP_TIME', 'CHASSIS_NUMBER'])  # Excluímos o alvo e o CHASSIS_NUMBER
y = df['LAID_UP_TIME']
print(f"Number of missing values in y: {y.isna().sum()}")

# Reservar uma parte para teste interno (20% do Dataset 1)
X_train, X_test_internal, y_train, y_test_internal = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar K-Fold Cross-Validation (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modelo (exemplo: Random Forest)
model = RandomForestRegressor(random_state=42)

# Armazenar resultados da validação cruzada
fold_scores = []

for train_index, val_index in kf.split(X_train):
    # Divisão dos folds
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Treinar o modelo
    model.fit(X_train_fold, y_train_fold)
    
    # Validar o modelo
    y_pred_val = model.predict(X_val_fold)
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
    fold_scores.append(fold_rmse)

# Calcular média e desvio padrão dos RMSE nos folds
mean_rmse = np.mean(fold_scores)
std_rmse = np.std(fold_scores)

# Avaliar no conjunto de teste interno
model.fit(X_train, y_train)  # Treinar com todos os dados de treino (excluindo o teste interno)
y_pred_test_internal = model.predict(X_test_internal)
test_internal_rmse = np.sqrt(mean_squared_error(y_test_internal, y_pred_test_internal))

# Resultados
print(f"RMSE médio na validação cruzada: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"RMSE no teste interno: {test_internal_rmse:.4f}")

# FINAL RESULTS
# Number of missing values in y: 0
# RMSE médio na validação cruzada: 47.0124 ± 15.4126
# RMSE no teste interno: 66.3128