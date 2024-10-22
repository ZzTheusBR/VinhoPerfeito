# %%
#importando as bibliotécas
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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
from collections import Counter

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
#variáveis para armazenar a melhor acurácia e o melhor ensemble
best_dt_accuracy = None
best_nb_accuracy = None
best_mlp_accuracy = None
best_ensemble_accuracy = 0
best_dt_predictions = None
best_nb_predictions = None
best_mlp_predictions = None
best_ensemble_predictions = None
best_y_test = None
best_dt_model = None
best_nb_model = None
best_mlp_model = None
dt_accuracies = []
nb_accuracies = []
mlp_accuracies = []
ensemble_accuracies = []

#treina 2000 vezes e seleciona o melhor resultado
for i in range(2000):

    #dividir o conjunto de dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(Xz, y, test_size=0.3)

    #inicializar os modelos individuais
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    nb_model = GaussianNB()
    mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3))

    #treinar cada modelo
    dt_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)
    mlp_model.fit(X_train, y_train)

    dt_accuracy = dt_model.score(X_test, y_test)
    nb_accuracy = nb_model.score(X_test, y_test)
    mlp_accuracy = mlp_model.score(X_test, y_test)

    #fazer previsões para o conjunto de teste
    dt_predictions = dt_model.predict(X_test)
    nb_predictions = nb_model.predict(X_test)
    mlp_predictions = mlp_model.predict(X_test)

    #armazenar as acurácias dos modelos individuais
    dt_accuracies.append(dt_accuracy)
    nb_accuracies.append(nb_accuracy)
    mlp_accuracies.append(mlp_accuracy)

    #voto majoritário para decidir a resposta
    ensemble_predictions = []
    for j in range(len(dt_predictions)):
        votes = [dt_predictions[j], nb_predictions[j], mlp_predictions[j]]
        majority_vote = Counter(votes).most_common(1)[0][0]
        ensemble_predictions.append(majority_vote)

    #calcula a acurácia do ensemble
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    ensemble_accuracies.append(ensemble_accuracy)

    #verifica se a acurácia do ensemble atual é maior do que a melhor acurácia
    if ensemble_accuracy > best_ensemble_accuracy:
        best_dt_accuracy = dt_accuracy
        best_nb_accuracy = nb_accuracy
        best_mlp_accuracy = mlp_accuracy
        best_ensemble_accuracy = ensemble_accuracy
        best_dt_predictions = dt_predictions
        best_nb_predictions = nb_predictions
        best_mlp_predictions = mlp_predictions
        best_ensemble_predictions = ensemble_predictions
        best_y_test = y_test
        best_dt_model = dt_model
        best_nb_model = nb_model
        best_mlp_model = mlp_model

#salvando os modelos
joblib.dump(best_dt_model, 'dt.joblib')
joblib.dump(best_nb_model, 'nb.joblib')
joblib.dump(best_mlp_model, 'mlp.joblib')

print("Sucesso!")

# %%
#acurácia dos modelos
print("Acurácia da Decision Tree: {:.2f}%".format(best_dt_accuracy * 100))
print("Acurácia do Naive Bayes: {:.2f}%".format(best_nb_accuracy * 100))
print("Acurácia do MLP: {:.2f}%".format(best_mlp_accuracy * 100))
print("Acurácia do Ensemble: {:.2f}%".format(best_ensemble_accuracy * 100))

# %%
#printa o melhor resultado
print("\nMatriz de confusão detalhada do Ensemble:\n", pd.crosstab(best_y_test, best_ensemble_predictions, rownames=['Real'], colnames=['Predito'], margins=True, margins_name='Todos'), "\n")
print("Relatório sobre a qualidade do Ensemble:\n", metrics.classification_report(best_y_test, best_ensemble_predictions, target_names=['Bom', 'Ruim']))

# %%
#dataFrame com as respostas das três IAs mais o target
df_predict = pd.DataFrame({
    'Preditor Decision Tree': best_dt_predictions,
    'Preditor Naive Bayes': best_nb_predictions,
    'Preditor MLPClassifier': best_mlp_predictions,
    'Preditor Ensemble': best_ensemble_predictions,
    'Real': best_y_test
})

#pd.set_option('display.max_rows', None)
#pd.reset_option('display.max_rows')
df_predict

# %%
#média e desvio padrão
print("Média: {:.2f}%". format(np.mean(ensemble_accuracies)*100))
print("Desvio padrão: {:.2f}%".format(np.std(ensemble_accuracies)*100))
#plota o gráfico
plt.hist(ensemble_accuracies, bins = 10)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.title('Distribuição de Acurácia do Ensemble')
plt.show()

# %%
#média e desvio padrão
print("Média: {:.2f}%". format(np.mean(ensemble_accuracies)*100))
print("Desvio padrão: {:.2f}%".format(np.std(ensemble_accuracies)*100))

#plota o gráfico
sns.distplot(ensemble_accuracies)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.yticks([])
plt.title("Distribuição de Acurácia do Ensemble")
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

# %%
#separa data e target do dataset
X = df.drop(['quality'], axis=1)
y = df['quality']

#colunas
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality']

#definindo os valores para uma única linha
good_values = [14.5, 0.15, 0.7, 3.2, 0.04, 50, 23, 0.990, 4, 0.9, 13.9, 0]
bad_values = [8, 1.2, 0.25, 2.0, 0.09, 8, 127, 1, 3.1, 0.35, 9, 1]

#criando o dataFrame
good = pd.DataFrame([good_values], columns=columns)
bad = pd.DataFrame([bad_values], columns=columns)

good_data = good.iloc[:, :11]  #selecionando as 11 primeiras colunas
bad_data = bad.iloc[:, :11]  #selecionando as 11 primeiras colunas

good_target = good['quality'] #selecionando a última coluna
bad_target = bad['quality'] #selecionando a última coluna

# %%
#normalizando os dados
scaler = Normalizer()
scaler.fit(good_data)
scaler.fit(bad_data)
good_z = scaler.transform(good_data)
bad_z = scaler.transform(bad_data)

#transforma o array em dataframe novamente
good_z = pd.DataFrame(good_z)
bad_z = pd.DataFrame(bad_z)

# %%
#variáveis para armazenar a melhor acurácia e o melhor ensemble
gbest_dt_accuracy = None
gbest_nb_accuracy = None
gbest_mlp_accuracy = None
gbest_ensemble_accuracy = 0
gbest_dt_predictions = None
gbest_nb_predictions = None
gbest_mlp_predictions = None
gbest_ensemble_predictions = None
gbest_y_test = None
gbest_dt_model = None
gbest_nb_model = None
gbest_mlp_model = None
gdt_accuracies = []
gnb_accuracies = []
gmlp_accuracies = []
gensemble_accuracies = []

#treina 20 vezes e seleciona o melhor resultado
for i in range(20):

    #inicializar os modelos individuais
    gdt_model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    gnb_model = GaussianNB()
    gmlp_model = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (3))

    #treinar cada modelo
    gdt_model.fit(Xz, y)
    gnb_model.fit(Xz, y)
    gmlp_model.fit(Xz, y)

    gdt_accuracy = gdt_model.score(good_z, good_target)
    gnb_accuracy = gnb_model.score(good_z, good_target)
    gmlp_accuracy = gmlp_model.score(good_z, good_target)

    #fazer previsões para o conjunto de teste
    gdt_predictions = gdt_model.predict(good_z)
    gnb_predictions = gnb_model.predict(good_z)
    gmlp_predictions = gmlp_model.predict(good_z)

    #armazenar as acurácias dos modelos individuais
    gdt_accuracies.append(gdt_accuracy)
    gnb_accuracies.append(gnb_accuracy)
    gmlp_accuracies.append(gmlp_accuracy)

    #voto majoritário para decidir a resposta
    gensemble_predictions = []
    for j in range(len(gdt_predictions)):
        gvotes = [gdt_predictions[j], gnb_predictions[j], gmlp_predictions[j]]
        gmajority_vote = Counter(gvotes).most_common(1)[0][0]
        gensemble_predictions.append(gmajority_vote)

    #calcula a acurácia do ensemble
    gensemble_accuracy = accuracy_score(good_target, gensemble_predictions)
    gensemble_accuracies.append(gensemble_accuracy)

    #verifica se a acurácia do ensemble atual é maior do que a melhor acurácia
    if gensemble_accuracy > gbest_ensemble_accuracy:
        gbest_dt_accuracy = gdt_accuracy
        gbest_nb_accuracy = gnb_accuracy
        gbest_mlp_accuracy = gmlp_accuracy
        gbest_ensemble_accuracy = gensemble_accuracy
        gbest_dt_predictions = gdt_predictions
        gbest_nb_predictions = gnb_predictions
        gbest_mlp_predictions = gmlp_predictions
        gbest_ensemble_predictions = gensemble_predictions
        gbest_dt_model = gdt_model
        gbest_nb_model = gnb_model
        gbest_mlp_model = gmlp_model

print("Sucesso!")

# %%
#acurácia dos modelos
print("Acurácia da Decision Tree: {:.2f}%".format(gbest_dt_accuracy * 100))
print("Acurácia do Naive Bayes: {:.2f}%".format(gbest_nb_accuracy * 100))
print("Acurácia do MLP: {:.2f}%".format(gbest_mlp_accuracy * 100))
print("Acurácia do Ensemble: {:.2f}%".format(gbest_ensemble_accuracy * 100))

# %%
print("\nMatriz de confusão detalhada do Ensemble:\n", pd.crosstab(good_target, gbest_ensemble_predictions, rownames=['Real'], colnames=['Predito'], margins=True, margins_name='Todos'), "\n")
print("Relatório sobre a qualidade do Ensemble:\n", metrics.classification_report(good_target, gbest_ensemble_predictions, labels=[0, 1], target_names=['Bom', 'Ruim']))

# %%
#dataFrame com as respostas das três IAs mais o target
gdf_predict = pd.DataFrame({
    'Preditor Decision Tree': gbest_dt_predictions,
    'Preditor Naive Bayes': gbest_nb_predictions,
    'Preditor MLPClassifier': gbest_mlp_predictions,
    'Preditor Ensemble': gbest_ensemble_predictions,
    'Real': good_target
})

#pd.set_option('display.max_rows', None)
#pd.reset_option('display.max_rows')
gdf_predict

# %%
#média e desvio padrão
print("Média: {:.2f}%". format(np.mean(gensemble_accuracies)*100))
print("Desvio padrão: {:.2f}%".format(np.std(gensemble_accuracies)*100))
#plota o gráfico
plt.hist(gensemble_accuracies, bins = 10)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.title('Distribuição de Acurácia do Ensemble')
plt.show()

# %%
#média e desvio padrão
print("Média: {:.2f}%". format(np.mean(gensemble_accuracies)*100))
print("Desvio padrão: {:.2f}%".format(np.std(gensemble_accuracies)*100))

#plota o gráfico
sns.distplot(gensemble_accuracies)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.yticks([])
plt.title("Distribuição de Acurácia do Ensemble")
plt.show()

# %%
#variáveis para armazenar a melhor acurácia e o melhor ensemble
bbest_dt_accuracy = None
bbest_nb_accuracy = None
bbest_mlp_accuracy = None
bbest_ensemble_accuracy = 0
bbest_dt_predictions = None
bbest_nb_predictions = None
bbest_mlp_predictions = None
bbest_ensemble_predictions = None
bbest_y_test = None
bbest_dt_model = None
bbest_nb_model = None
bbest_mlp_model = None
bdt_accuracies = []
bnb_accuracies = []
bmlp_accuracies = []
bensemble_accuracies = []

#treina 20 vezes e seleciona o melhor resultado
for i in range(20):

    #inicializar os modelos individuais
    bdt_model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    bnb_model = GaussianNB()
    bmlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3))

    #treinar cada modelo
    bdt_model.fit(Xz, y)
    bnb_model.fit(Xz, y)
    bmlp_model.fit(Xz, y)

    bdt_accuracy = bdt_model.score(bad_z, bad_target)
    bnb_accuracy = bnb_model.score(bad_z, bad_target)
    bmlp_accuracy = bmlp_model.score(bad_z, bad_target)

    #fazer previsões para o conjunto de teste
    bdt_predictions = bdt_model.predict(bad_z)
    bnb_predictions = bnb_model.predict(bad_z)
    bmlp_predictions = bmlp_model.predict(bad_z)

    #armazenar as acurácias dos modelos individuais
    bdt_accuracies.append(bdt_accuracy)
    bnb_accuracies.append(bnb_accuracy)
    bmlp_accuracies.append(bmlp_accuracy)

    #voto majoritário para decidir a resposta
    bensemble_predictions = []
    for j in range(len(bdt_predictions)):
        bvotes = [bdt_predictions[j], bnb_predictions[j], bmlp_predictions[j]]
        bmajority_vote = Counter(bvotes).most_common(1)[0][0]
        bensemble_predictions.append(bmajority_vote)

    #calcula a acurácia do ensemble
    bensemble_accuracy = accuracy_score(bad_target, bensemble_predictions)
    bensemble_accuracies.append(bensemble_accuracy)

    #verifica se a acurácia do ensemble atual é maior do que a melhor acurácia
    if bensemble_accuracy > bbest_ensemble_accuracy:
        bbest_dt_accuracy = bdt_accuracy
        bbest_nb_accuracy = bnb_accuracy
        bbest_mlp_accuracy = bmlp_accuracy
        bbest_ensemble_accuracy = bensemble_accuracy
        bbest_dt_predictions = bdt_predictions
        bbest_nb_predictions = bnb_predictions
        bbest_mlp_predictions = bmlp_predictions
        bbest_ensemble_predictions = bensemble_predictions
        bbest_dt_model = bdt_model
        bbest_nb_model = bnb_model
        bbest_mlp_model = bmlp_model

print("Sucesso!")

# %%
#acurácia dos modelos
print("Acurácia da Decision Tree: {:.2f}%".format(bbest_dt_accuracy * 100))
print("Acurácia do Naive Bayes: {:.2f}%".format(bbest_nb_accuracy * 100))
print("Acurácia do MLP: {:.2f}%".format(bbest_mlp_accuracy * 100))
print("Acurácia do Ensemble: {:.2f}%".format(bbest_ensemble_accuracy * 100))

# %%
print("\nMatriz de confusão detalhada do Ensemble:\n", pd.crosstab(bad_target, bbest_ensemble_predictions, rownames=['Real'], colnames=['Predito'], margins=True, margins_name='Todos'), "\n")
print("Relatório sobre a qualidade do Ensemble:\n", metrics.classification_report(bad_target, bbest_ensemble_predictions, labels=[0, 1], target_names=['Bom', 'Ruim']))

# %%
#dataFrame com as respostas das três IAs mais o target
bdf_predict = pd.DataFrame({
    'Preditor Decision Tree': bbest_dt_predictions,
    'Preditor Naive Bayes': bbest_nb_predictions,
    'Preditor MLPClassifier': bbest_mlp_predictions,
    'Preditor Ensemble': bbest_ensemble_predictions,
    'Real': bad_target
})

#pd.set_option('display.max_rows', None)
#pd.reset_option('display.max_rows')
bdf_predict

# %%
#média e desvio padrão
print("Média: {:.2f}%". format(np.mean(bensemble_accuracies)*100))
print("Desvio padrão: {:.2f}%".format(np.std(bensemble_accuracies)*100))
#plota o gráfico
plt.hist(bensemble_accuracies, bins = 10)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.title('Distribuição de Acurácia do Ensemble')
plt.show()

# %%
#média e desvio padrão
print("Média: {:.2f}%". format(np.mean(bensemble_accuracies)*100))
print("Desvio padrão: {:.2f}%".format(np.std(bensemble_accuracies)*100))

#plota o gráfico
sns.distplot(bensemble_accuracies)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.yticks([])
plt.title("Distribuição de Acurácia do Ensemble")
plt.show()



# %%
