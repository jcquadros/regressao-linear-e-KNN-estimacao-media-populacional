import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Passo 1: Carregue os dados do CSV
data = pd.read_csv('seuarquivo.csv')  # Substitua 'seuarquivo.csv' pelo nome do seu arquivo CSV

# Passo 2: Separe o conjunto em treino e teste
X = data[['X1', 'X2', 'X3', 'X4']]
Y = data['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Passo 3: Ajuste modelos KNN e regressão linear e calcule o erro de previsão
k_values = [1, 3, 5]
errors = {'KNN_1': [], 'KNN_3': [], 'KNN_5': [], 'LinearRegression': []}

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    errors[f'KNN_{k}'] = np.abs(Y_test - Y_pred)

lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
errors['LinearRegression'] = np.abs(Y_test - Y_pred)

# Passo 4: Estime o erro absoluto médio (MAE) para cada método
mae_results = {}
for method, error in errors.items():
    mae = np.mean(error)
    mae_std = np.std(error)
    mae_results[method] = {'MAE': mae, 'MAE_std': mae_std}

# Passo 5: Conclusões sobre os resultados
for method, result in mae_results.items():
    print(f'{method}: MAE = {result["MAE"]:.2f} (Desvio Padrão = {result["MAE_std"]:.2f})')

# Você também pode comparar os modelos com base nos resultados de MAE e escolher o melhor método.
