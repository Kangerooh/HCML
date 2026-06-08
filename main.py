from folktables import ACSDataSource, ACSIncome
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# data loading

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA", "NY", "TX"], download=True) # only take 3 states since datasize will get too big otherwise

feature_names = ACSIncome.features
df = acs_data[feature_names].copy()
df['label'] = (acs_data['PINCP'] > 50000).astype(int)

# Recode SEX to 0/1 already
df['SEX'] = df['SEX'].map({1: 0, 2: 1})  # 0=Male, 1=Female

def preprocessing(df):
    # Check missing originally
    print("Missing values per column:")
    print(df.isnull().sum())
    print(f"\nTotal missing: {df.isnull().sum().sum()}")

    # work from copy from now on + drop nans
    df_clean = df.copy()
    df_clean = df_clean.dropna()
    print(f"\nOriginal size: {len(df)} | After dropping NaNs: {len(df_clean)}")

    continuous_features  = ['AGEP', 'WKHP']
    protected_attributes = ['SEX', 'RAC1P']   # kept separate for fairness eval
    categorical_features = [col for col in df.columns
                            if col not in continuous_features + protected_attributes + ['label', 'age_group']]

    print("Continuous: ", continuous_features)

    print("Categorical:", categorical_features)
    print("Protected:  ", protected_attributes)

    return df_clean, continuous_features, categorical_features, protected_attributes

def data_split(df):
    X = df.drop(columns=['label', 'age_group'], errors='ignore')
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Train positive rate: {y_train.mean():.3f} | Test positive rate: {y_test.mean():.3f}")

    # Save protected attributes separately for fairness evaluation later
    sex_test  = X_test['SEX'].values
    race_test = X_test['RAC1P'].values

    return X_train, X_test, y_train, y_test, sex_test, race_test

def processing_pipeline(X_train, X_test, continuous_features, categorical_features, protected_attributes):
    # Keep SEX and RAC1P as passthrough (not transformed)
    # so we can easily include/exclude them in model configurations
    preprocessor = ColumnTransformer(transformers=[
        ('scale', StandardScaler(), continuous_features),
        ('ohe',   OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('protect', 'passthrough', protected_attributes)  # keep as-is
    ])

    # Fit on train, transform both
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Recover feature names for readability
    ohe_feature_names = preprocessor.named_transformers_['ohe'].get_feature_names_out(categorical_features)
    all_feature_names = continuous_features + list(ohe_feature_names) + protected_attributes

    X_train_df = pd.DataFrame(X_train_proc, columns=all_feature_names)
    X_test_df  = pd.DataFrame(X_test_proc,  columns=all_feature_names)


    print(f"Processed feature matrix shape: {X_train_df.shape}")
    print(X_train_df.head(2))

    return X_train_df, X_test_df

def main():
    df_clean, continuous_features, categorical_features, protected_attributes = preprocessing(df)
    X_train, X_test, y_train, y_test, sex_test, race_test = data_split(df_clean)
    X_train_df, X_test_df = processing_pipeline(X_train, X_test, continuous_features, categorical_features, protected_attributes)

    # Config 1: all features (including SEX and RAC1P)
    X_train_all = X_train_df.copy()
    X_test_all  = X_test_df.copy()

    # Config 2: without SEX (unawareness)
    X_train_nosex = X_train_df.drop(columns=['SEX'])
    X_test_nosex  = X_test_df.drop(columns=['SEX'])

    # Config 3: without proxy features: DO THIS after the proxy identification
    # X_train_noproxy =

    print("Config 1 shape:", X_train_all.shape)
    print("Config 2 shape:", X_train_nosex.shape)

if __name__ == "__main__":
    main()