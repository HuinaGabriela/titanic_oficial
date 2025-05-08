import pandas as pd

def load_data(train_path, test_path):
    """Carrega os dados de treino e teste a partir dos caminhos."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def fill_missing_values(df):
    """Preenche valores ausentes nas colunas Age, Fare e Embarked."""
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

def add_family_size(df):
    """Cria a coluna FamilySize baseada em SibSp e Parch."""
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def extract_title(df):
    """Extrai o título do nome e agrupa os títulos raros."""
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df
