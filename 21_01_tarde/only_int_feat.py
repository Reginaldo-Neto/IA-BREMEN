import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
df_combined = df_combined.drop(columns=["RPAKREP_VEHICLE_HKEY", "MANUFACTURER", "MODEL_CODE", "OPERATING_HOURS", 
 "MILAGE_IN_FIELD", "OPERATING_HOURS_SALES", "RIM_KEY", "COLOR_CODE", 
 "COLOR_CODE_NAME", "COLOR_TYPE", "UPHOLSTERY_CODE", "UPHOLSTERY_CODE_ALT", 
 "CERTIFICATE_TYPE", "CERTIFICATE_TYPE_DATE", "FACTORY_NUMBER", "ENGINE_ID", 
 "ENGINE_ID_ALT", "TRANSMISSION", "TRANSMISSION_ID", "RIMS", "FRONT_TIRES", 
 "FRONT_TIRES_CONDITION", "REAR_TIRES", "REAR_TIRES_CONDITION", 
 "PERMITTED_TOTAL_WEIGHT", "MAX_TRAILOR_LOAD", "REPAIR_RKZ", 
 "OPTICAL_CONDITION", "TECHNICAL_CONDITION", "COMMISSION_NUMBER", 
 "LEASING_CONTRACT_DATE", "LEASING_START", "LEASING_END", "PAINT_TYPE", 
 "FINANCING_TYPE_NAME", "FUEL_TYPE", "DRIVE_TYPE_NAME", "VEHICLE_MODEL_ID_NAME", 
 "COMMISSION_TYPE_NAME", "DEMONSTRATION_STATUS", "PURCHASE_DATE", 
 "PURCHASE_BOOKING_DATE", "PURCHASE_OPERATION_HOURS", "PRICE_LIST", 
 "DAY_OF_REGISTRATION", "SOLD_CUSTOMER_ID", "SOLD_INVOICE_COSTUMER_ID", 
 "OPERATION_HOURS_SALE", "SOLD_INVOICE_COSTUMER_ID2", "CUSTOMER_GROUP_NAME", 
 "CUSTOMER_FEATURE_NAME", "SALE_CUSTOMER_ID2", "CUSTOMER_SALE_GROUP_NAME", 
 "CUSTOMER_SALE_GROUP2_NAME"])

# Manter apenas colunas numéricas e 'CHASSIS_NUMBER'
num_cols = df_combined.select_dtypes(include=["float64", "int64"]).columns.tolist()
df_combined = df_combined[num_cols + ["CHASSIS_NUMBER"]]

# Preencher valores ausentes no 'CHASSIS_NUMBER' para evitar erros (substituir por 'UNKNOWN')
df_combined["CHASSIS_NUMBER"] = df_combined["CHASSIS_NUMBER"].fillna("UNKNOWN")

# Separar novamente os conjuntos
df_train_encoded = df_combined.iloc[:len(df_train)]
df_test_encoded = df_combined.iloc[len(df_train):]
target = "LAID_UP_TIME"
# Limpar linhas com `LAID_UP_TIME` vazio no conjunto de treino
df_train_encoded = df_train_encoded.dropna(subset=[target])

# Separar variáveis explicativas e alvo
X = df_train_encoded.drop(columns=[target])
y = df_train_encoded[target]
X_test = df_test_encoded.drop(columns=[target], errors="ignore")

# Verificar e tratar valores ausentes
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Divisão 70-15-15 no conjunto de treino
X_train_full, X_temp, y_train_full, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test_final, y_val, y_test_final = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_full, y_train_full)

# Avaliação no conjunto de validação final
y_val_pred = rf_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE no conjunto de validação final: {val_rmse:.2f}")

# Previsões no subconjunto de teste interno do treino
y_test_internal_pred = rf_model.predict(X_test_final)
internal_test_results = pd.DataFrame({
    "CHASSIS_NUMBER": df_train_encoded.iloc[X_test_final.index]["CHASSIS_NUMBER"],
    "Real": y_test_final,
    "Previsto": y_test_internal_pred,
    "Erro": y_test_final - y_test_internal_pred
})
internal_test_results.to_excel("internal_test_predictions.xlsx", index=False)
print("Resultados do subconjunto de teste interno salvos em 'internal_test_predictions.xlsx'.")

# Previsões no conjunto de teste final
y_test_pred = rf_model.predict(X_test)
final_test_results = pd.DataFrame({
    "CHASSIS_NUMBER": df_test_encoded["CHASSIS_NUMBER"],
    "Previsto": y_test_pred
})
final_test_results.to_excel("final_test_predictions.xlsx", index=False)
print("Previsões para o conjunto de teste final salvas em 'final_test_predictions.xlsx'.")
