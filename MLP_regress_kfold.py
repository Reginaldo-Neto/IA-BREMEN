import joblib
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Carregar o Dataset 1 (limpo)
dataset1_path = "cleaned_train_dataset_v2.xlsx"
df = pd.read_excel(dataset1_path)

# Excluir linhas com valores vazios na coluna 'LAID_UP_TIME'
df = df.dropna(subset=['LAID_UP_TIME'])

# Preencher valores NaN com 0 em todos os atributos restantes
df = df.fillna(0)

# Separar features (X) e target (y)
X = df.drop(columns=['LAID_UP_TIME', 'CHASSIS_NUMBER'])  # Excluímos o alvo e o CHASSIS_NUMBER
y = df['LAID_UP_TIME']

# Reservar uma parte para teste interno (20% do Dataset 1)
X_train, X_test_internal, y_train, y_test_internal = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar K-Fold Cross-Validation (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modelo (exemplo: Random Forest)
model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42, learning_rate='adaptive', early_stopping=True, activation='relu', solver='adam')

# Armazenar resultados da validação cruzada
fold_scores = []

for train_index, val_index in kf.split(X_train):
    print(f"Train index: {train_index}")
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
model.fit(X_train, y_train)
y_pred_mlp = model.predict(X_test_internal)
rmse_mlp = np.sqrt(mean_squared_error(y_test_internal, y_pred_mlp))
print(f"RMSE com MLP: {rmse_mlp:.4f}")

# Obter importâncias das features
feature_importances = model.feature_importances_

# Criar um DataFrame com as features e suas importâncias
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Salvar em um arquivo de texto
feature_importance_filename = "feature_importance_MLP.txt"
feature_importance_df.to_csv(feature_importance_filename, index=False, sep="\t")
print(f"Importâncias das features salvas em: {feature_importance_filename}")

# Salvar o modelo treinado
model_filename = "MLP_model.pkl"
joblib.dump(model, model_filename)
print(f"Modelo salvo como: {model_filename}")

# FINAL RESULTS
# RMSE com MLP: 75.7557