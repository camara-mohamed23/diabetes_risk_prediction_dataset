# ==========================================================
# DIABETES RISK PREDICTION
# CHARGEMENT DES DONNÉES
# ==========================================================

import pandas as pd

from config import DATA_FILE


# ==========================================================
# CHARGEMENT DU DATASET
# ==========================================================

def load_data():

    print("=" * 70)
    print("CHARGEMENT DU DATASET")
    print("=" * 70)

    print(f"Fichier : {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    print(
        f"Dataset chargé : "
        f"{df.shape[0]:,} lignes × "
        f"{df.shape[1]} colonnes"
    )

    return df


# ==========================================================
# NETTOYAGE DES NOMS DE COLONNES
# ==========================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
    )

    return df


# ==========================================================
# NETTOYAGE DES VALEURS TEXTE
# ==========================================================

def clean_text_columns(df):

    df = df.copy()

    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    return df


# ==========================================================
# NETTOYAGE COMPLET
# ==========================================================

def clean_data(df):

    print("\n" + "=" * 70)
    print("NETTOYAGE DES DONNÉES")
    print("=" * 70)

    df = clean_column_names(df)

    df = clean_text_columns(df)

    # Suppression des lignes complètement vides
    before = len(df)

    df = df.dropna(
        how="all"
    )

    after = len(df)

    print(
        "Lignes complètement vides supprimées :",
        before - after
    )

    # Suppression des doublons
    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        "Doublons supprimés :",
        before - after
    )

    return df


# ==========================================================
# RAPPORT DATASET
# ==========================================================

def dataset_report(df):

    print("\n" + "=" * 70)
    print("RAPPORT DU DATASET")
    print("=" * 70)

    print("\nDimensions :")
    print(df.shape)

    print("\nValeurs manquantes :")

    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        print("Aucune valeur manquante.")

    else:

        print(missing)

    print("\nTypes de données :")

    print(df.dtypes)

    print("\nStatistiques numériques :")

    print(df.describe().T)


# ==========================================================
# TEST DIRECT
# ==========================================================

if __name__ == "__main__":

    df = load_data()

    df = clean_data(df)

    dataset_report(df)