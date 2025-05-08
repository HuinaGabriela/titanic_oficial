# Importações e leitura dos dados
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Importa as funções de src/data_preprocessing.py
from src.data_preprocessing import (
    load_data,
    fill_missing_values,
    extract_title,
    add_family_size
)

# Pré-processamento dos dados
train_df, test_df = load_data("data/raw/train.csv", "data/raw/test.csv")

train_df = fill_missing_values(train_df)
train_df = extract_title(train_df)
train_df = add_family_size(train_df)

test_df = fill_missing_values(test_df)
test_df = extract_title(test_df)
test_df = add_family_size(test_df)

# Preparação dos dados (get_dummies e alinhamento)
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]
# get_dummies_converte variáveis categóricas em variáveis numéricas
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])

# Alinhar colunas
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

y = train_df["Survived"]

# Definição dos hiperparâmetros e busca
# Parâmetros para RandomizedSearch
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 5, 10]
}

model = XGBClassifier(eval_metric='logloss', random_state=42)

# Validação estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, 
    n_iter=30, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1, random_state=42
)

random_search.fit(X, y)

# Mostrar melhores parâmetros e acurácia
print("Melhores parâmetros encontrados:")
print(random_search.best_params_)

print("\nMelhor acurácia (validação cruzada):")
print(round(random_search.best_score_, 4))

# VISUALIZAÇÃO (gráficos baseados apenas no train_data original, não diretamente no modelo final otimizado com XGBoost)

# 👩 Mulheres sobreviveram muito mais que homens.

# 💰 Passageiros da 1ª classe tiveram maior taxa de sobrevivência.

# 🚢 Passageiros que embarcaram em Cherbourg (C) sobreviveram mais.

# 👨‍👩‍👧‍👦 Famílias pequenas (2–4 membros) têm melhores chances.

# 🧑 Títulos como Miss, Mrs, Master têm taxas mais altas que Mr.

sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

# Par de (nome da feature, título interno do gráfico)
plot_config = [
    ('Sex', 'Sexo'),
    ('Pclass', 'Classe'),
    ('Embarked', 'Porto de Embarque'),
    ('FamilySize', 'Tamanho da Família'),
    ('Title', 'Título')
]

# Gráficos
for i, (feature, label) in enumerate(plot_config):
    order = train_df[feature].value_counts().index if feature == 'Title' else None
    sns.countplot(x=feature, hue='Survived', data=train_df, ax=axes[i], order=order)

    # Título dentro do gráfico
    axes[i].text(0.02, 0.95, f'Sobrevivência por {label}',
                 transform=axes[i].transAxes,
                 fontsize=12, fontweight='bold', va='top')

    # Rotação para títulos longos
    if feature == 'Title':
        axes[i].tick_params(axis='x', rotation=45)

# Remove slot vazio
fig.delaxes(axes[5])

# Ajuste de espaçamento
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.show()

# Gráfico mostra quais colunas criadas (após get_dummies) foram mais relevantes para o modelo final.
best_model = random_search.best_estimator_
# Importância das features no modelo XGBoost final
plt.figure(figsize=(10, 6))
importances = best_model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Importância das Features no Modelo XGBoost")
plt.xlabel("Importância")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Salva o melhor modelo treinado
joblib.dump(best_model, 'models/best_xgb_model.pkl')
print("Modelo salvo em 'models/best_xgb_model.pkl'")

# Gerar submissão final com o melhor modelo
best_model = random_search.best_estimator_

# Prever no conjunto de teste
predictions = best_model.predict(X_test)

# Salvar submissão
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('submission_xgboost_tuned.csv', index=False)

print("\nSubmissão salva como 'submission_xgboost_tuned.csv'.")
