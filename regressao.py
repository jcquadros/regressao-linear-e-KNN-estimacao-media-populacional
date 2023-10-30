import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('Trabalho.csv')  

# Separando o conjunto em treino e teste
X = data[['X1', 'X2', 'X3', 'X4']]
Y = data['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Ajustando modelos KNN e regressão linear e calcule o erro de previsão
k_values = [1, 3, 5]
errors = {'KNN_1': [], 'KNN_3': [], 'KNN_5': [], 'Regressão_Linear': []}

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    errors[f'KNN_{k}'] = np.abs(Y_test - Y_pred)

lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
errors['Regressão_Linear'] = np.abs(Y_test - Y_pred)

# Estimando o erro absoluto médio para cada método
erro_results = {}
for method, error in errors.items():
    erro_medio = np.mean(error)
    erro_std = np.std(error)
    erro_results[method] = {'erro absoluto médio': erro_medio, 'desvio padrao': erro_std}

# ResultadosS
for method, result in erro_results.items():
    print(f'{method}: erro absoluto médio = {result["erro absoluto médio"]:.2f} (Desvio Padrão = {result["desvio padrao"]:.2f})')

