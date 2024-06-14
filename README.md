# sam

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Baixar dados históricos de ações do Bradesco
ticker = 'BBDC4.SA'
data = yf.download(ticker, start='2013-01-01', end='2023-01-01')

# Usar a coluna 'Close' para o modelo ARIMA
closing_prices = data['Close']

# Plotar os dados
closing_prices.plot(figsize=(10, 6))
plt.title(f'{ticker} Preços de Fechamento')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.show()

# Dividir os dados em treino e teste
train_data = closing_prices[:int(0.8 * len(closing_prices))]
test_data = closing_prices[int(0.8 * len(closing_prices)):]

# Treinar o modelo ARIMA
model = ARIMA(train_data, order=(5, 1, 0))  # Ordem (p, d, q)
fitted_model = model.fit()

# Fazer previsões
forecast = fitted_model.forecast(steps=len(test_data))

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data, label='Dados de Treino')
plt.plot(test_data.index, test_data, label='Dados de Teste')
plt.plot(test_data.index, forecast, label='Previsão', color='red')
plt.title(f'{ticker} Previsão ARIMA')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.legend()
plt.show()

# Mostrar a previsão
print(forecast)

