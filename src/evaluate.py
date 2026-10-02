# ==========================================================
# DIABETES RISK PREDICTION
# MODEL EVALUATION
# ==========================================================

import os

import joblib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from data_loader import (
    load_data,
    clean_data
)

from preprocessing import (
    remove_unused_columns,
    split_features_target
)

from config import (
    MODEL_FILE,
    TARGET,
    REPORTS_DIR,
    RANDOM_STATE,
    TEST_SIZE
)


# ==========================================================
# DOSSIER
# ==========================================================

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_model():

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    df = load_data()

    df = clean_data(df)

    df = remove_unused_columns(df)

    X, y = split_features_target(df)

    # ------------------------------------------------------
    # Split
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    # ------------------------------------------------------
    # Chargement modèle
    # ------------------------------------------------------

    model = joblib.load(
        MODEL_FILE
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # ------------------------------------------------------
    # Classification report
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    # ------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------

    with open(
        REPORTS_DIR /
        "classification_report.txt",
        "w"
    ) as file:

        file.write(report)

    # ------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    labels = sorted(
        y.unique()
    )

    plt.figure(
        figsize=(9, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(
        "Matrice de confusion"
    )

    plt.xlabel(
        "Prédiction"
    )

    plt.ylabel(
        "Valeur réelle"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR /
        "confusion_matrix.png",
        dpi=150
    )

    plt.close()

    print(
        "\nMatrice de confusion sauvegardée."
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    evaluate_model()