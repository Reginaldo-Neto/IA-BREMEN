import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Carregar o Dataset 1 (limpo)
dataset1_path = "cleaned_train_dataset_v2.xlsx"
df = pd.read_excel(dataset1_path)

# Excluir linhas com valores vazios na coluna 'LAID_UP_TIME'
df = df.dropna(subset=['LAID_UP_TIME'])
# Lista das 40 colunas mais importantes
important_columns = [
    "SCALED_INVENTURAL_VALUE", "COMMISSION_TYPE", "MILAGE_SALES", "SCALED_REPORT_VALUE",
    "PURCHASE_MILAGE", "MILAGE_SALE", "VEHICLE_TYPE", "CURB_WEIGHT",
    "YEAR_CONSTRUCTION", "SCALED_GUIDE_PRICE", "VEHICLE_MODEL_ID", "NUMBER_SEATS",
    "SCALED_CURRENT_VALUE", "SCALED_TOTAL_SALES_PRICE_BASIS", "DRIVE_TYPE", "NUMBER_AXLE",
    "COMPANY", "UPHOLSTERY", "MANUFACTURER_SHORT", "NUMBER_DOORS",
    "OFFICE_MAIN_BRAND", "LEASING_MILAGE", "SCALED_TOTAL_SALE_PRICE", "FINANCING_TYPE",
    "OFFICE", "TRANSMISSION_TYPE", "TRANSMISSION_NAME", "FUEL_TYPE_NAME", "ACCIDENT_VEHICLE",
    "NUMBER_OWNERS", "KW", "HORSEPOWER", "VEHICLE_GROUP", "COLOR", "ENGINE_TYPE",
    "CUSTOMER_SALE_GROUP", "CUSTOMER_SALE_GROUP2", "CCM", "MILEAGE", "NUMBER_ENGINE_CYLINDER", 'LAID_UP_TIME', 'CHASSIS_NUMBER'
]

# Manter apenas as colunas importantes no DataFrame
df = df[important_columns]

# Separar features (X) e target (y)
X = df.drop(columns=['LAID_UP_TIME', 'CHASSIS_NUMBER'])  # Excluímos o alvo e o CHASSIS_NUMBER
y = df['LAID_UP_TIME']
# Reservar uma parte para teste interno (20% do Dataset 1)
X_train, X_test_internal, y_train, y_test_internal = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar K-Fold Cross-Validation (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = XGBRegressor(random_state=42)

# Armazenar resultados da validação cruzada
fold_scores = []

for train_index, val_index in kf.split(X_train):
    print(f"Train index: {train_index}")
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

model.fit(X_train, y_train)
y_pred_xgb = model.predict(X_test_internal)
rmse_xgb = np.sqrt(mean_squared_error(y_test_internal, y_pred_xgb))
print(f"RMSE com XGBoost no teste dropado: {rmse_xgb:.4f}")

# Obter importâncias das features
feature_importances = model.feature_importances_

# Criar um DataFrame com as features e suas importâncias
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Salvar em um arquivo de texto
feature_importance_filename = "feature_importance_dropada.txt"
feature_importance_df.to_csv(feature_importance_filename, index=False, sep="\t")
print(f"Importâncias das features salvas em: {feature_importance_filename}")

# Salvar o modelo treinado
model_filename = "XGB_model_dropado.pkl"
joblib.dump(model, model_filename)
print(f"Modelo salvo como: {model_filename}")

# FINAL RESULTS
# RMSE com XGBoost no teste: 68.1718