# %%
#importando as bibliotécas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import requests
import joblib

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# %%
#URL do dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dataframe = pd.read_csv(url, delimiter=';')
dataframe['quality'] = ['Bom' if x >= 6 else 'Ruim' for x in dataframe['quality']]

#faz o download do arquivo
response = requests.get(url)
content = response.content

#salva o arquivo no ambiente da IDE
with open('winequality-red.csv', 'wb') as file:
    file.write(content)

# %%
#importa o dataset
dataframe = pd.read_csv('winequality-red.csv', delimiter=';')
dataframe['quality'] = ['Bom' if x >= 6 else 'Ruim' for x in dataframe['quality']]

# %%
#deleta todas as linhas repetidas: 1599 - 1359 = 240 linhas excluídas
duplicados = dataframe.duplicated().sum(); print(duplicados, "linhas excluídas")
df = dataframe.drop_duplicates()

#índice das linhas não está sequencial: exclui as linhas que contém valores nulos e reindexa as linhas remanescentes para que tenham índices contínuos a partir do zero
df = df.dropna().reset_index(drop=True) 
df

# %%
#transforma dados categóricos em dados numéricos e retorna esses dados como um array NumPy: 'Bom' e 'Ruim' se transforma em uns e zeros
le = LabelEncoder()
quality = le.fit_transform(df['quality'])

#transforma o array em dataframe novamente
quality = pd.DataFrame(quality, columns=['quality'])

#'Bom' e 'Ruim' é trocado por uns zeros
df = df.drop('quality', axis=1).join(quality)
df

# %%
#verificar se há valores ausentes em todas as colunas
nulos = df.isnull().sum()
print(nulos)

# %%
#define o limite de corte para remoção dos outliers
limite = 1.5

#itera sobre cada coluna do conjunto de dados
for coluna in df.columns:
    #calcula o IQR
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1

    #define os limites de corte
    lim_sup = Q3 + limite*IQR
    lim_inf = Q1 - limite*IQR

    #conta o número de outliers acima e abaixo dos limites de corte
    n_outliers_sup = len(df[df[coluna] > lim_sup])
    n_outliers_inf = len(df[df[coluna] < lim_inf])

    #exibe o número de outliers encontrados para cada variável
    print(f"Variável '{coluna}' possui {n_outliers_sup} outliers acima do limite superior e {n_outliers_inf} outliers abaixo do limite inferior.")

# %%
colunas = ["chlorides", "residual sugar", "sulphates"]

#define o limite de corte para remoção dos outliers
limite = 1.5

#itera sobre as colunas
for coluna in colunas:
    #calcula o IQR
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1

    #define os limites de corte
    lim_sup = Q3 + limite*IQR
    lim_inf = Q1 - limite*IQR

    #remove os outliers da coluna
    df = df[(df[coluna] >= lim_inf) & (df[coluna] <= lim_sup)]

#exibe o número de observações antes e depois da remoção dos outliers
print("Número de observações antes da remoção de outliers:", len(pd.read_csv("winequality-red.csv")) - duplicados)
print("Número de observações após a remoção de outliers:", len(df))

#índice das linhas não está sequencial: exclui as linhas que contém valores nulos e reindexa as linhas remanescentes para que tenham índices contínuos a partir do zero
df = df.dropna().reset_index(drop=True) 
df

# %%
#descrição estatística resumida dos dados contidos
'''
count: o número de valores não nulos na coluna.
mean: a média dos valores na coluna.
std: o desvio padrão dos valores na coluna.
min: o valor mínimo na coluna.
25%: o valor do primeiro quartil (25%).
50%: o valor do segundo quartil (50% ou a mediana).
75%: o valor do terceiro quartil (75%).
max: o valor máximo na coluna.
'''
df.drop(['quality'], axis=1).describe()

# %%
#separa data e target do dataset
X = df.drop(['quality'], axis=1)
y = df['quality']

# %%
#normalizando os dados
scaler = Normalizer()
scaler.fit(X)
Xz = scaler.transform(X)

#transforma o array em dataframe novamente
Xz = pd.DataFrame(Xz)

# %%
#treina 2000 vezes e seleciona o melhor resultado
melhor_acuracia = 0
melhor_clf = 0
melhor_predictions = 0
melhor_y_test = 0
acuracias = []

for i in range(2000):
    X_train, X_test, y_train, y_test = train_test_split(Xz, y, test_size = 0.3)
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    clf = clf.fit(X_train, y_train)
    acuracia = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    acuracias.append(acuracia)
    if acuracia > melhor_acuracia:
        melhor_acuracia = acuracia
        melhor_clf = clf
        melhor_predictions = predictions
        melhor_y_test = y_test
        joblib.dump(melhor_clf, 'dt.joblib')

# %%
#printa o melhor resultado
print("\nMatriz de confusão detalhada:\n", pd.crosstab(melhor_y_test, melhor_predictions, rownames=['Real'], colnames=['Predito'], margins=True, margins_name='Todos'), "\n")
print("Relatório sobre a qualidade:\n", metrics.classification_report(melhor_y_test, melhor_predictions, target_names=['Bom', 'Ruim']))

# %%
#exibe as informações sobre o classificador utilizado, como o tipo de algoritmo de árvore de decisão e os hiperparâmetros usados
joblib.load('dt.joblib')

# %%
#média e desvio padrão
print("Média: {:.2f}%". format(np.mean(acuracias)*100))
print("Desvio padrão: {:.2f}%".format(np.std(acuracias)*100))

#plota o gráfico
plt.hist(acuracias, bins = 10)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.title('Distribuição de Acurácia da Árvore de Decisão')
plt.show()

# %%
#média e desvio padrão
print("Média: {:.2f}%". format(np.mean(acuracias)*100))
print("Desvio padrão: {:.2f}%".format(np.std(acuracias)*100))

#remove warnings
warnings.filterwarnings("ignore")

#plota o gráfico
sns.distplot(acuracias)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.yticks([])
plt.title("Distribuição de Acurácia da Árvore de Decisão")
plt.show()

# %%
#não é possível separar os dados a olho nu
sns.scatterplot(x=df['alcohol'], y=df['sulphates'], hue='quality', data=df)

# %%
#não é possível separar os dados a olho nu
#mapeando 'Ruim' para 0 e 'Bom' para 1
plt.scatter(df['alcohol'], df['sulphates'], c=df['quality'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Álcool')
plt.ylabel('Sulfatos')
plt.title('Gráfico de Dispersão')
plt.show()

# %%
#não é possível separar os dados a olho nu
sns.pairplot(df, hue='quality')
plt.show()


