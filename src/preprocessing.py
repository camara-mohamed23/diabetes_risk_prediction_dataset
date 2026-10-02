

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer

from config import (
    TARGET,
    LEAKAGE_COLUMNS
)


# ==========================================================
# SUPPRESSION DES COLONNES INUTILES
# ==========================================================

def remove_unused_columns(df):

    df = df.copy()

    columns_to_remove = [
        column
        for column in LEAKAGE_COLUMNS
        if column in df.columns
    ]

    if columns_to_remove:

        print("\nColonnes supprimées :")

        for column in columns_to_remove:
            print(f" - {column}")

        df = df.drop(
            columns=columns_to_remove
        )

    return df


# ==========================================================
# SÉPARATION FEATURES / TARGET
# ==========================================================

def split_features_target(df):

    if TARGET not in df.columns:

        raise ValueError(
            f"La colonne cible '{TARGET}' "
            f"n'existe pas dans le dataset."
        )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    return X, y


# ==========================================================
# IDENTIFICATION DES VARIABLES
# ==========================================================

def identify_columns(X):

    numeric_columns = (
        X.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32"
            ]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    return (
        numeric_columns,
        categorical_columns
    )


# ==========================================================
# CONSTRUCTION DU PREPROCESSOR
# ==========================================================

def build_preprocessor(
    numeric_columns,
    categorical_columns
):

    # ------------------------------------------------------
    # PIPELINE NUMÉRIQUE
    # ------------------------------------------------------
    #
    # 1. Remplacement des valeurs manquantes
    #    par la médiane
    #
    # 2. Standardisation
    #
    # ------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    # ------------------------------------------------------
    # PIPELINE CATÉGORIEL
    # ------------------------------------------------------
    #
    # 1. Remplacement des valeurs manquantes
    #    par la modalité la plus fréquente
    #
    # 2. One-Hot Encoding
    #
    # ------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


    # ------------------------------------------------------
    # COMBINAISON
    # ------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],

        remainder="drop"
    )

    return preprocessor


# ==========================================================
# RAPPORT DU PREPROCESSING
# ==========================================================

def preprocessing_report(
    X,
    numeric_columns,
    categorical_columns
):

    print("\n" + "=" * 70)
    print("VARIABLES DU MODÈLE")
    print("=" * 70)

    print(
        "\nNombre de variables numériques :",
        len(numeric_columns)
    )

    for column in numeric_columns:

        print(
            f"  - {column}"
        )

    print(
        "\nNombre de variables catégorielles :",
        len(categorical_columns)
    )

    for column in categorical_columns:

        print(
            f"  - {column}"
        )


# ==========================================================
# RAPPORT DES VALEURS MANQUANTES
# ==========================================================

def missing_values_report(df):

    print("\n" + "=" * 70)
    print("VALEURS MANQUANTES")
    print("=" * 70)

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        print(
            "Aucune valeur manquante."
        )

    else:

        for column, count in missing.items():

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"{column:<40} "
                f"{count:>8,} "
                f"({percentage:.2f}%)"
            )


# ==========================================================
# TEST DIRECT
# ==========================================================

if __name__ == "__main__":

    from data_loader import (
        load_data,
        clean_data
    )

    df = load_data()

    df = clean_data(df)

    missing_values_report(df)

    df = remove_unused_columns(df)

    X, y = split_features_target(df)

    (
        numeric_columns,
        categorical_columns
    ) = identify_columns(X)

    preprocessing_report(
        X,
        numeric_columns,
        categorical_columns
    )

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    print("\nPreprocessor créé avec succès.")
