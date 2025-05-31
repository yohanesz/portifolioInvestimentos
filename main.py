import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ativos = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'B3SA3.SA', 'WEGE3.SA']

dados = yf.download(ativos, start='2022-01-01', end='2024-12-31')['Close']
dados = dados.dropna()

retornos = dados.pct_change().dropna()
    
media_retorno = retornos.mean()
cov_matriz = retornos.cov()

NUM_ATIVOS = len(ativos)
POPULACAO = 80
GERACOES = 300  
TAXA_MUTACAO = 0.02
NUM_ILHAS = 3
MIGRACAO_FREQ = 50  
MIGRANTES = 3       

def avaliar_fitness(individuo):
    retorno = np.dot(individuo, media_retorno) * 252
    risco = np.sqrt(np.dot(individuo.T, np.dot(cov_matriz * 252, individuo)))
    return retorno / risco if risco != 0 else 0

def inicializar_populacao():
    return np.array([np.random.dirichlet(np.ones(NUM_ATIVOS)) for _ in range(POPULACAO)])

def selecionar_pais(populacao, fitness):
    idx = np.random.choice(range(POPULACAO), size=3, replace=False)
    melhor = idx[np.argmax(fitness[idx])]
    return populacao[melhor]

def crossover(pai1, pai2):
    alpha = np.random.uniform(0, 1)
    filho = alpha * pai1 + (1 - alpha) * pai2
    return filho / filho.sum()

def mutar(individuo):
    for i in range(NUM_ATIVOS):
        if np.random.rand() < TAXA_MUTACAO:
            individuo[i] += np.random.normal(0, 0.05)
    individuo = np.abs(individuo)
    return individuo / individuo.sum()


def algoritmo_genetico():
    populacao = inicializar_populacao()
    historico = []

    for geracao in range(GERACOES):
        fitness = np.array([avaliar_fitness(ind) for ind in populacao])
        nova_populacao = []

        for _ in range(POPULACAO):
            pai1 = selecionar_pais(populacao, fitness)
            pai2 = selecionar_pais(populacao, fitness)
            filho = crossover(pai1, pai2)
            filho = mutar(filho)
            nova_populacao.append(filho)

        populacao = np.array(nova_populacao)

        melhor_idx = np.argmax(fitness)
        historico.append(fitness[melhor_idx])

    melhor_idx = np.argmax(fitness)
    melhor = populacao[melhor_idx]
    return melhor, historico

# 🌍 AG com Ilhas
def algoritmo_genetico_ilhas():
    ilhas = [inicializar_populacao() for _ in range(NUM_ILHAS)]
    historico = []

    for geracao in range(GERACOES):
        for i in range(NUM_ILHAS):
            fitness = np.array([avaliar_fitness(ind) for ind in ilhas[i]])
            nova_populacao = []

            for _ in range(POPULACAO):
                pai1 = selecionar_pais(ilhas[i], fitness)
                pai2 = selecionar_pais(ilhas[i], fitness)
                filho = crossover(pai1, pai2)
                filho = mutar(filho)
                nova_populacao.append(filho)

            ilhas[i] = np.array(nova_populacao)

        # 🌐 Migração entre ilhas
        if (geracao + 1) % MIGRACAO_FREQ == 0:
            for i in range(NUM_ILHAS):
                fonte = ilhas[i]
                destino = ilhas[(i + 1) % NUM_ILHAS]
                fitness_fonte = np.array([avaliar_fitness(ind) for ind in fonte])
                melhores_indices = fitness_fonte.argsort()[-MIGRANTES:]
                migrantes = fonte[melhores_indices]
                destino[:MIGRANTES] = migrantes

        # 🔥 Melhor fitness entre as ilhas
        melhor_fitness = max(
            np.max([avaliar_fitness(ind) for ind in ilha]) for ilha in ilhas
        )
        historico.append(melhor_fitness)

    # 🔍 Melhor indivíduo global
    melhor_individuo = None
    melhor_fitness = -np.inf
    for ilha in ilhas:
        fitness = np.array([avaliar_fitness(ind) for ind in ilha])
        idx = np.argmax(fitness)
        if fitness[idx] > melhor_fitness:
            melhor_fitness = fitness[idx]
            melhor_individuo = ilha[idx]

    return melhor_individuo, historico

# ⚙️ Cálculo Manual (linha constante)
pesos_manual = np.array([1/NUM_ATIVOS]*NUM_ATIVOS)
retorno_manual = np.dot(pesos_manual, media_retorno) * 252
risco_manual = np.sqrt(np.dot(pesos_manual.T, np.dot(cov_matriz * 252, pesos_manual)))
sharpe_manual = retorno_manual / risco_manual
historico_manual = [sharpe_manual] * GERACOES

# ▶️ Executar os algoritmos
melhor_tradicional, historico_tradicional = algoritmo_genetico()
melhor_ilhas, historico_ilhas = algoritmo_genetico_ilhas()

# 📈 Plotar gráfico de comparação
plt.figure(figsize=(10,6))
plt.plot(historico_tradicional, label='AG Tradicional', color='red')
plt.plot(historico_ilhas, label='AG com Ilhas', color='green')
plt.plot(historico_manual, label='Manual (Constante)', color='blue', linestyle='dashed')

plt.title('Evolução do Índice de Sharpe')
plt.xlabel('Geração')
plt.ylabel('Índice de Sharpe')
plt.legend()
plt.grid(True)
plt.show()
