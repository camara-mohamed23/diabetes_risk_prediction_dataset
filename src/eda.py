# ==========================================================
# DIABETES RISK PREDICTION
# EXPLORATORY DATA ANALYSIS
# ==========================================================

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import (
    load_data,
    clean_data
)

from config import (
    REPORTS_DIR,
    TARGET
)


# ==========================================================
# CRÉATION DU DOSSIER REPORTS
# ==========================================================

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ==========================================================
# ANALYSE DE LA CIBLE
# ==========================================================

def analyze_target(df):

    print("\n" + "=" * 70)
    print("ANALYSE DE LA VARIABLE CIBLE")
    print("=" * 70)

    counts = (
        df[TARGET]
        .value_counts()
    )

    percentages = (
        df[TARGET]
        .value_counts(
            normalize=True
        )
        * 100
    )

    result = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print(result)

    plt.figure(
        figsize=(10, 6)
    )

    sns.countplot(
        data=df,
        x=TARGET
    )

    plt.title(
        "Distribution du risque de diabète"
    )

    plt.xlabel(
        "Diabetes Risk"
    )

    plt.ylabel(
        "Nombre de patients"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR /
        "target_distribution.png",
        dpi=150
    )

    plt.close()


# ==========================================================
# HISTOGRAMMES
# ==========================================================

def numerical_distributions(df):

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
    )

    for column in numerical_columns:

        plt.figure(
            figsize=(9, 5)
        )

        sns.histplot(
            data=df,
            x=column,
            kde=True
        )

        plt.title(
            f"Distribution de {column}"
        )

        plt.tight_layout()

        filename = (
            REPORTS_DIR /
            f"distribution_{column}.png"
        )

        plt.savefig(
            filename,
            dpi=120
        )

        plt.close()


# ==========================================================
# BOXPLOTS PAR RAPPORT AU RISQUE
# ==========================================================

def risk_boxplots(df):

    numerical_columns = [
        column
        for column in df.select_dtypes(
            include=np.number
        ).columns
        if column != "Patient_ID"
    ]

    for column in numerical_columns:

        plt.figure(
            figsize=(10, 6)
        )

        sns.boxplot(
            data=df,
            x=TARGET,
            y=column
        )

        plt.title(
            f"{column} selon le risque de diabète"
        )

        plt.xticks(
            rotation=30
        )

        plt.tight_layout()

        filename = (
            REPORTS_DIR /
            f"risk_{column}.png"
        )

        plt.savefig(
            filename,
            dpi=120
        )

        plt.close()


# ==========================================================
# MATRICE DE CORRÉLATION
# ==========================================================

def correlation_matrix(df):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
    )

    correlation = numeric_df.corr()

    plt.figure(
        figsize=(18, 14)
    )

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Matrice de corrélation"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR /
        "correlation_matrix.png",
        dpi=150
    )

    plt.close()

    return correlation


# ==========================================================
# ANALYSE CATÉGORIELLE
# ==========================================================

def categorical_analysis(df):

    categorical_columns = (
        df.select_dtypes(
            include=["object"]
        )
        .columns
    )

    for column in categorical_columns:

        if column == TARGET:
            continue

        plt.figure(
            figsize=(10, 6)
        )

        sns.countplot(
            data=df,
            x=column,
            hue=TARGET
        )

        plt.title(
            f"{column} selon le risque"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        filename = (
            REPORTS_DIR /
            f"category_{column}.png"
        )

        plt.savefig(
            filename,
            dpi=120
        )

        plt.close()


# ==========================================================
# PROGRAMME PRINCIPAL
# ==========================================================

def main():

    df = load_data()

    df = clean_data(df)

    analyze_target(df)

    numerical_distributions(df)

    risk_boxplots(df)

    correlation_matrix(df)

    categorical_analysis(df)

    print(
        "\nEDA terminée."
    )

    print(
        f"Rapports disponibles dans : "
        f"{REPORTS_DIR}"
    )


if __name__ == "__main__":

    main()