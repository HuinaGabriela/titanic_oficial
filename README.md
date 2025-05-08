Kaggle · Competição de aprendizado de máquina a partir de desastres.

# 🛳️ Titanic - Machine Learning from Disaster

Este projeto é uma solução para o clássico desafio do [Kaggle Titanic](https://www.kaggle.com/competitions/titanic) usando machine learning para prever quais passageiros sobreviveram ao naufrágio com base em dados disponíveis.

## 📂 Estrutura do Projeto
```
titanic_oficial/
│
├── data/ # Dados crus (train.csv, test.csv)
├── src/
│ ├── init.py # Torna src um pacote
│ └── data_preprocessing.py # Funções de pré-processamento
├── models/ # Modelos treinados (.pkl/.joblib)
├── main.py # Pipeline principal de treino e submissão
├── requirements.txt # Bibliotecas necessárias
├── submission_xgboost_tuned.csv # Submissão para o Kaggle
└── README.md
```
## ⚙️ Tecnologias Utilizadas
- Python 3.10
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost
- Joblib

## 📊 Abordagem

1. **Pré-processamento de dados** (`src/data_preprocessing.py`):
   - Preenchimento de valores ausentes
   - Extração de títulos dos nomes (`Mr`, `Mrs`, etc.)
   - Criação de nova feature: `FamilySize`
   - One-hot encoding (`get_dummies`)
   - Alinhamento entre treino e teste

2. **Treinamento do modelo**:
   - Uso de `XGBClassifier`
   - Validação cruzada com `StratifiedKFold`
   - Busca de hiperparâmetros com `RandomizedSearchCV`

3. **Visualizações**:
   - Gráficos com `seaborn` para análise de sobrevivência por variáveis como sexo, classe, título, etc.
   - Gráfico de importância de features baseado no XGBoost

4. **Exportação**:
   - Geração de `submission.csv`
   - Salvamento do modelo final com `joblib`

## 📈 Resultados
- Modelo final: `XGBoost`
- Métrica de avaliação: `accuracy` com validação cruzada
- Arquivo gerado: `submission_xgboost_tuned.csv`
- Score no Kaggle: Melhor acurácia (validação cruzada):~0.8507


