# sam

#Passo 1: Importar a base de dados
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Carregar dados históricos
# Substitua 'historical_stock_prices.csv' pelo caminho do seu arquivo CSV
data = pd.read_csv('Clientes.csv', index_col='Date', parse_dates=True)

print(data)

# Dividir os dados em treino e teste

stock_prices = data['Close']

train_size = int(len(stock_prices) * 0.8)
train, test = stock_prices[:train_size], stock_prices[train_size:]

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Ajustar o modelo ARIMA
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()


# Verifique o nome da coluna de data e preços
date_column = 'Date'
price_column = 'Close'

# Verificar se há valores nulos na coluna de preços
print(data[price_column].isnull().sum())

# Converter a coluna de data para o tipo datetime se não estiver no formato datetime
data[date_column] = pd.to_datetime(data[date_column])

# Verificar se há valores nulos na coluna de preços e removê-los
print(f"Valores nulos na coluna '{price_column}':", data[price_column].isnull().sum())
data = data.dropna(subset=[price_column])

# Verificar se a coluna de preços contém apenas valores numéricos
data[price_column] = pd.to_numeric(data[price_column], errors='coerce')

# Remover linhas onde a conversão para numérico resultou em NaN
data = data.dropna(subset=[price_column])

# Carregar dados históricos com a coluna de data como índice
data = data.set_index(date_column)

# Certificar-se de que a coluna de preços está no tipo float
data[price_column] = data[price_column].astype(float)

# Selecionar a coluna de preços
stock_prices = data[price_column]

# Fazer previsões
predictions = model_fit.forecast(steps=len(test))

# Plotar os resultados
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Predicted')
plt.legend()
plt.show()

# Avaliação
rmse = np.sqrt(np.mean((predictions - test)**2))
print(f'RMSE: {rmse}')
