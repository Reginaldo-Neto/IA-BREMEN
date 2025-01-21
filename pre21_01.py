import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Carregar os arquivos de treino e teste
train_path = "Data/Vehicles_export_prices_scaled_train_eng.xlsx"  # Substitua pelo caminho correto
test_path = "Data/Vehicles_export_prices_scaled_stud_test_eng.xlsx"  # Substitua pelo caminho correto
train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

# Concatenar os datasets com marcador de origem
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Converter colunas de data para dias desde a época
for col in ['LEASING_CONTRACT_DATE', 'LEASING_START', 'LEASING_END', 'PURCHASE_DATE', 'PURCHASE_BOOKING_DATE']:
    combined_df[col] = (pd.to_datetime(combined_df[col]) - pd.Timestamp("1970-01-01")).dt.days

# Preservar CHASSIS_NUMBER e as últimas 7 colunas
chassis_column = combined_df['CHASSIS_NUMBER']
normalized_cols = list(combined_df.columns[-7:])
combined_df = combined_df.drop(columns=['CHASSIS_NUMBER'])

# Identificar colunas categóricas e aplicar Label Encoding
categorical_cols = combined_df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))
    label_encoders[col] = le
    
# Identificar colunas numéricas (excluindo as últimas 7, 'SCALED_*' e 'LAID_UP_TIME')
numerical_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
numerical_cols = [
    col for col in numerical_cols 
    if col not in normalized_cols and not col.startswith('SCALED') and col != 'LAID_UP_TIME'
]

# Normalizar colunas numéricas
scaler = MinMaxScaler()
combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])

# Restaurar CHASSIS_NUMBER e preservar as últimas 7 colunas do dataset combinado
combined_df['CHASSIS_NUMBER'] = chassis_column
for col in normalized_cols:
    combined_df[col] = pd.concat([train_df[col], test_df[col]], ignore_index=True)

# Separar os datasets de volta em treino e teste
train_cleaned = combined_df[combined_df['dataset'] == 'train'].drop(columns=['dataset'])
test_cleaned = combined_df[combined_df['dataset'] == 'test'].drop(columns=['dataset'])

# Salvar os datasets processados
train_cleaned.to_excel("cleaned_train_dataset_v2.xlsx", index=False)
test_cleaned.to_excel("cleaned_test_dataset_v2.xlsx", index=False)
