# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:12:47 2025

@author: MATHEUSLOPESMARTINS
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1. Coleta e Preparação de Dados (Exemplo Simulado) ---
# Em um cenário real, você buscaria dados de APIs como CoinGecko, CryptoCompare, etc.
# Para este exemplo, vamos simular alguns dados históricos de uma criptomoeda.

# Gerar datas de 1º de janeiro de 2020 até hoje (5 de julho de 2025)
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 7, 5) # Data atual
dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Simular preços com uma tendência crescente e alguma volatilidade
# Isso é uma simplificação ENORME para fins de exemplo
np.random.seed(42) # Para reprodutibilidade
base_price = 100 # Preço inicial base
trend = np.linspace(0, 500, len(dates)) # Tendência linear crescente
volatility = np.random.normal(0, 30, len(dates)) # Volatilidade aleatória
prices = base_price + trend + volatility
# Garantir que os preços não sejam negativos, caso a volatilidade os leve para baixo
prices = np.maximum(prices, 10)

# Criar um DataFrame
df = pd.DataFrame({'Date': dates, 'Price': prices})
df['Date'] = pd.to_datetime(df['Date'])

# Converter a data para um formato numérico (número de dias desde uma data de referência)
# Isso é necessário para que o modelo de regressão linear possa processar a data.
df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days

# --- 2. Preparar os Dados para o Modelo ---
X = df[['Days_Since_Start']] # Variável independente (tempo)
y = df['Price']             # Variável dependente (preço)

# --- 3. Escolher e Treinar o Modelo (Regressão Linear) ---
model = LinearRegression()
model.fit(X, y)

# --- 4. Fazer a Previsão até 2030 ---
# Gerar datas futuras até o final de 2030
future_start_date = df['Date'].max() + timedelta(days=1)
future_end_date = datetime(2030, 12, 31)
future_dates = [future_start_date + timedelta(days=x) for x in range((future_end_date - future_start_date).days + 1)]

# Converter as datas futuras para o mesmo formato numérico
future_df = pd.DataFrame({'Date': future_dates})
future_df['Days_Since_Start'] = (future_df['Date'] - df['Date'].min()).dt.days

# Fazer as previsões
future_predictions = model.predict(future_df[['Days_Since_Start']])

# --- 5. Visualizar os Resultados ---
plt.figure(figsize=(14, 7))

# Plotar dados históricos
plt.plot(df['Date'], df['Price'], label='Preço Histórico', color='blue')

# Plotar a linha de regressão (ajustada aos dados históricos)
# Isso mostra a tendência que o modelo "aprendeu"
plt.plot(df['Date'], model.predict(X), label='Tendência do Modelo (Dados Históricos)', color='green', linestyle='--')

# Plotar previsões futuras
plt.plot(future_df['Date'], future_predictions, label='Previsão até 2030', color='red', linestyle='-.')

# Formatar o eixo X para datas
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate() # Rotação automática das datas

plt.title('Previsão Simplificada de Preço de Criptomoeda até 2030 (Regressão Linear)')
plt.xlabel('Data')
plt.ylabel('Preço (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Imprimir o preço estimado no final de 2030
estimated_price_2030_end = future_predictions[-1]
print(f"Preço estimado para 31 de dezembro de 2030: ${estimated_price_2030_end:,.2f}")