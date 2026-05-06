import re
import pickle
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import Config


class Dataset:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_checkpoint = {}   # хранит медианы, моды и прочие fit-значения

    """Обработка данных: заполнение пропусков, feature engineering, кодирование."""
    def get_dataset(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

        y_train = df_train[self.config.target_col].copy()

        df_train = self._fill_missing(df_train, fit=True)
        df_test  = self._fill_missing(df_test,  fit=False)

        df_train = self._feature_engineering(df_train)
        df_test  = self._feature_engineering(df_test)

        df_train = self._extract_title(df_train, fit=True)
        df_test  = self._extract_title(df_test,  fit=False)

        df_train, df_test = self._encode_categoricals(df_train, df_test)

        drop = [self.config.target_col, self.config.id_col,
                'Name', 'Ticket', 'Cabin']
        X_train = df_train.drop(columns=[c for c in drop if c in df_train.columns])
        X_test  = df_test.drop(columns=[c for c in drop if c in df_test.columns])

        # Выравниваем колонки test по train (OHE может добавить разные столбцы)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        print(f'X_train: {X_train.shape} | X_test: {X_test.shape}')
        print(f'Признаки: {X_train.columns.tolist()}')
        assert X_train.isnull().sum().sum() == 0, 'В X_train остались пропуски!'
        assert X_test.isnull().sum().sum()  == 0, 'В X_test остались пропуски!'

        return X_train, X_test, y_train


    def _fill_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        # Age
        if fit:
            self.data_checkpoint['age_group_median'] = (
                df.groupby(['Pclass', 'Sex'])['Age'].median()
            )
        group_median = self.data_checkpoint['age_group_median']
        df['Age'] = df.apply(
            lambda r: group_median[r['Pclass'], r['Sex']]
            if pd.isna(r['Age']) else r['Age'],
            axis=1
        )

        # Embarked
        if fit:
            self.data_checkpoint['embarked_mode'] = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(self.data_checkpoint['embarked_mode'])

        # Fare — медиана по классу из train + замена нулей
        if fit:
            self.data_checkpoint['fare_medians'] = (
                df.groupby('Pclass')['Fare'].median()
            )
        fare_medians = self.data_checkpoint['fare_medians']
        bad_fare = (df['Fare'] == 0) | df['Fare'].isna()
        df.loc[bad_fare, 'Fare'] = df.loc[bad_fare, 'Pclass'].map(fare_medians)

        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
        df['IsChild']    = (df['Age'] < 16).astype(int)
        df['Deck']       = df['Cabin'].str[0].fillna('U')

        return df

    def _extract_title(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        df['Title'] = df['Name'].str.extract(r', ([A-Za-z]+)\.')
        df['Title'] = df['Title'].replace(
            {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
        )

        if fit:
            rare = df['Title'].value_counts()[
                df['Title'].value_counts() < 10
            ].index
            self.data_checkpoint['rare_titles'] = list(rare)

        df['Title'] = df['Title'].replace(
            self.data_checkpoint['rare_titles'], 'Rare'
        )

        return df

    def _encode_categoricals(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cats_cols_encoding = {'Sex': 'label', 'Embarked': 'onehot', 'Deck': 'label', 'Title': 'label'}

        for col, enc in cats_cols_encoding.items():
            if enc == 'onehot':
                combined = pd.concat(
                    [df_train[col], df_test[col]], axis=0
                )
                combined_enc = pd.get_dummies(combined)

                train_enc = combined_enc.iloc[:len(df_train)].reset_index(drop=True)
                test_enc  = combined_enc.iloc[len(df_train):].reset_index(drop=True)

                df_train = pd.concat([
                    df_train.drop(columns=[col]).reset_index(drop=True),
                    train_enc
                ], axis=1)
                df_test = pd.concat([
                    df_test.drop(columns=[col]).reset_index(drop=True),
                    test_enc
                ], axis=1)

            elif enc == 'label':
                le = LabelEncoder()
                le.fit(pd.concat([df_train[col], df_test[col]], axis=0).astype(str))
                df_train[col] = le.transform(df_train[col].astype(str))
                df_test[col]  = le.transform(df_test[col].astype(str))

        return df_train, df_test
