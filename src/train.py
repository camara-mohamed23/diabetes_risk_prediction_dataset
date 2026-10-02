
# ==========================================================
# DIABETES RISK PREDICTION
# MACHINE LEARNING - TRAINING COMPLET
# ==========================================================
#
# Objectifs :
#
# 1. Charger le dataset
# 2. Nettoyer les données
# 3. Supprimer les colonnes pouvant provoquer une fuite
# 4. Séparer X et y
# 5. Encoder la variable cible :
#       Low      -> 0
#       Moderate -> 1
#       High     -> 2
#
# 6. Séparer Train / Test
# 7. Préparer les variables numériques
# 8. Préparer les variables catégorielles
# 9. Gérer les valeurs manquantes
# 10. Standardiser les variables numériques
# 11. Encoder les variables catégorielles
# 12. Entraîner plusieurs modèles
# 13. Comparer les performances
# 14. Effectuer une Cross-Validation
# 15. Sauvegarder le meilleur modèle
# 16. Sauvegarder le LabelEncoder
#
# ==========================================================


# ==========================================================
# 1. IMPORTS
# ==========================================================

import os

import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ==========================================================
# 2. IMPORTS DU PROJET
# ==========================================================

from data_loader import (
    load_data,
    clean_data
)

from preprocessing import (
    remove_unused_columns,
    split_features_target,
    identify_columns,
    build_preprocessor
)

from config import (
    MODEL_FILE,
    METRICS_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS
)


# ==========================================================
# 3. CRÉATION DES DOSSIERS
# ==========================================================

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ==========================================================
# 4. FICHIER LABEL ENCODER
# ==========================================================

LABEL_ENCODER_FILE = (
    MODELS_DIR /
    "label_encoder.pkl"
)


# ==========================================================
# 5. FONCTION PRINCIPALE
# ==========================================================

def train_models():

    # ======================================================
    # ÉTAPE 1
    # CHARGEMENT DU DATASET
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 1 - CHARGEMENT DU DATASET")
    print("=" * 70)

    df = load_data()


    # ======================================================
    # ÉTAPE 2
    # NETTOYAGE
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 2 - NETTOYAGE DES DONNÉES")
    print("=" * 70)

    df = clean_data(df)


    # ======================================================
    # ÉTAPE 3
    # SUPPRESSION DES COLONNES INUTILES
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 3 - SUPPRESSION DES COLONNES")
    print("=" * 70)

    df = remove_unused_columns(df)


    # ======================================================
    # ÉTAPE 4
    # SÉPARATION FEATURES / TARGET
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 4 - FEATURES / TARGET")
    print("=" * 70)

    X, y = split_features_target(df)

    print(
        f"Nombre de variables explicatives : "
        f"{X.shape[1]}"
    )

    print(
        f"Nombre d'observations : "
        f"{X.shape[0]:,}"
    )


    # ======================================================
    # ÉTAPE 5
    # ANALYSE DE LA VARIABLE CIBLE
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 5 - VARIABLE CIBLE")
    print("=" * 70)

    print("\nClasses originales :")

    print(
        y.value_counts()
    )

    print("\nPourcentages :")

    print(
        (
            y.value_counts(
                normalize=True
            ) * 100
        ).round(2)
    )


    # ======================================================
    # ÉTAPE 6
    # ENCODAGE DE LA TARGET
    # ======================================================
    #
    # XGBoost attend des classes numériques.
    #
    # Exemple :
    #
    # High      -> 0
    # Low       -> 1
    # Moderate  -> 2
    #
    # L'ordre exact dépend de LabelEncoder.
    #
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 6 - ENCODAGE DE LA VARIABLE CIBLE")
    print("=" * 70)

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y
    )

    print("\nCorrespondance des classes :")

    for encoded_value, original_value in enumerate(
        label_encoder.classes_
    ):

        print(
            f"  {original_value} -> "
            f"{encoded_value}"
        )


    # ======================================================
    # ÉTAPE 7
    # IDENTIFICATION DES VARIABLES
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 7 - IDENTIFICATION DES VARIABLES")
    print("=" * 70)

    (
        numeric_columns,
        categorical_columns
    ) = identify_columns(X)

    print(
        "\nVariables numériques :",
        len(numeric_columns)
    )

    print(
        "Variables catégorielles :",
        len(categorical_columns)
    )


    # ======================================================
    # AFFICHAGE DES VARIABLES NUMÉRIQUES
    # ======================================================

    print("\nVariables numériques :")

    for column in numeric_columns:

        print(
            f"  - {column}"
        )


    # ======================================================
    # AFFICHAGE DES VARIABLES CATÉGORIELLES
    # ======================================================

    print("\nVariables catégorielles :")

    for column in categorical_columns:

        print(
            f"  - {column}"
        )


    # ======================================================
    # ÉTAPE 8
    # CONSTRUCTION DU PREPROCESSOR
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 8 - PREPROCESSING")
    print("=" * 70)

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    print(
        "Preprocessor créé avec succès."
    )


    # ======================================================
    # ÉTAPE 9
    # TRAIN / TEST SPLIT
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 9 - TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,

            y_encoded,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y_encoded
        )
    )

    print(
        "\nTrain :",
        X_train.shape
    )

    print(
        "Test :",
        X_test.shape
    )


    # ======================================================
    # DISTRIBUTION TRAIN
    # ======================================================

    print("\nDistribution TRAIN :")

    unique_train, counts_train = np.unique(
        y_train,
        return_counts=True
    )

    for class_id, count in zip(
        unique_train,
        counts_train
    ):

        class_name = (
            label_encoder
            .inverse_transform(
                [class_id]
            )[0]
        )

        print(
            f"  {class_name:<10} : "
            f"{count:,}"
        )


    # ======================================================
    # DISTRIBUTION TEST
    # ======================================================

    print("\nDistribution TEST :")

    unique_test, counts_test = np.unique(
        y_test,
        return_counts=True
    )

    for class_id, count in zip(
        unique_test,
        counts_test
    ):

        class_name = (
            label_encoder
            .inverse_transform(
                [class_id]
            )[0]
        )

        print(
            f"  {class_name:<10} : "
            f"{count:,}"
        )


    # ======================================================
    # ÉTAPE 10
    # DÉFINITION DES MODÈLES
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 10 - MODÈLES MACHINE LEARNING")
    print("=" * 70)


    models = {

        # --------------------------------------------------
        # LOGISTIC REGRESSION
        # --------------------------------------------------

        "Logistic Regression":

            LogisticRegression(

                max_iter=2000,

                random_state=RANDOM_STATE
            ),


        # --------------------------------------------------
        # RANDOM FOREST
        # --------------------------------------------------

        "Random Forest":

            RandomForestClassifier(

                n_estimators=300,

                max_depth=None,

                min_samples_split=2,

                random_state=RANDOM_STATE,

                n_jobs=-1
            ),


        # --------------------------------------------------
        # GRADIENT BOOSTING
        # --------------------------------------------------

        "Gradient Boosting":

            GradientBoostingClassifier(

                n_estimators=200,

                learning_rate=0.05,

                max_depth=3,

                random_state=RANDOM_STATE
            ),


        # --------------------------------------------------
        # XGBOOST
        # --------------------------------------------------

        "XGBoost":

            XGBClassifier(

                n_estimators=300,

                learning_rate=0.05,

                max_depth=6,

                subsample=0.8,

                colsample_bytree=0.8,

                objective="multi:softprob",

                num_class=len(
                    label_encoder.classes_
                ),

                eval_metric="mlogloss",

                random_state=RANDOM_STATE,

                n_jobs=-1
            )
    }


    # ======================================================
    # LISTES POUR STOCKER LES RÉSULTATS
    # ======================================================

    results = []

    trained_models = {}


    # ======================================================
    # ÉTAPE 11
    # ENTRAÎNEMENT DES MODÈLES
    # ======================================================

    for name, model in models.items():

        print("\n")
        print("=" * 70)
        print(
            f"ENTRAÎNEMENT : {name}"
        )
        print("=" * 70)


        # ==================================================
        # PIPELINE COMPLET
        # ==================================================
        #
        # Données
        #    ↓
        # Imputation
        #    ↓
        # Scaling / Encoding
        #    ↓
        # Modèle
        #
        # ==================================================

        pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )


        # ==================================================
        # TRAINING
        # ==================================================

        pipeline.fit(

            X_train,

            y_train
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        y_pred = pipeline.predict(
            X_test
        )


        # ==================================================
        # PROBABILITÉS
        # ==================================================

        roc_auc = np.nan

        if hasattr(
            pipeline,
            "predict_proba"
        ):

            try:

                y_probability = (
                    pipeline.predict_proba(
                        X_test
                    )
                )

                roc_auc = roc_auc_score(

                    y_test,

                    y_probability,

                    multi_class="ovr",

                    average="weighted"
                )

            except Exception as error:

                print(
                    "\nROC-AUC non disponible :",
                    error
                )


        # ==================================================
        # MÉTRIQUES
        # ==================================================

        accuracy = accuracy_score(

            y_test,

            y_pred
        )


        precision = precision_score(

            y_test,

            y_pred,

            average="weighted",

            zero_division=0
        )


        recall = recall_score(

            y_test,

            y_pred,

            average="weighted",

            zero_division=0
        )


        f1 = f1_score(

            y_test,

            y_pred,

            average="weighted",

            zero_division=0
        )


        # ==================================================
        # CROSS VALIDATION
        # ==================================================

        print(
            "\nCross-validation "
            f"({CV_FOLDS} folds)..."
        )

        cv_scores = cross_val_score(

            pipeline,

            X_train,

            y_train,

            cv=CV_FOLDS,

            scoring="f1_weighted",

            n_jobs=-1
        )


        cv_mean = cv_scores.mean()

        cv_std = cv_scores.std()


        # ==================================================
        # AFFICHAGE DES RÉSULTATS
        # ==================================================

        print(
            f"\nAccuracy : "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision : "
            f"{precision:.4f}"
        )

        print(
            f"Recall : "
            f"{recall:.4f}"
        )

        print(
            f"F1-score : "
            f"{f1:.4f}"
        )

        if not np.isnan(
            roc_auc
        ):

            print(
                f"ROC-AUC : "
                f"{roc_auc:.4f}"
            )

        else:

            print(
                "ROC-AUC : N/A"
            )

        print(
            f"CV F1 : "
            f"{cv_mean:.4f} "
            f"+/- {cv_std:.4f}"
        )


        # ==================================================
        # STOCKAGE DES RÉSULTATS
        # ==================================================

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1_Score": f1,

            "ROC_AUC": roc_auc,

            "CV_F1_Mean": cv_mean,

            "CV_F1_STD": cv_std
        })


        trained_models[name] = pipeline


    # ======================================================
    # ÉTAPE 12
    # COMPARAISON DES MODÈLES
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 12 - COMPARAISON DES MODÈLES")
    print("=" * 70)


    results_df = pd.DataFrame(
        results
    )


    # ------------------------------------------------------
    # Tri par F1-score
    # ------------------------------------------------------

    results_df = results_df.sort_values(

        by="F1_Score",

        ascending=False
    )


    # ------------------------------------------------------
    # Affichage
    # ------------------------------------------------------

    print(
        "\n"
        + results_df.to_string(
            index=False
        )
    )


    # ======================================================
    # ÉTAPE 13
    # SAUVEGARDE DES MÉTRIQUES
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 13 - SAUVEGARDE DES MÉTRIQUES")
    print("=" * 70)


    results_df.to_csv(

        METRICS_FILE,

        index=False
    )


    print(
        "Métriques sauvegardées :"
    )

    print(
        METRICS_FILE
    )


    # ======================================================
    # ÉTAPE 14
    # SÉLECTION DU MODÈLE
    # ======================================================
    #
    # Le modèle est sélectionné selon le F1-score
    # sur le jeu de test.
    #
    # Pour un projet plus rigoureux, nous ajouterons
    # ensuite une sélection basée sur la Cross-Validation.
    #
    # ======================================================

    best_model_name = (
        results_df
        .iloc[0]
        ["Model"]
    )


    best_model = (
        trained_models[
            best_model_name
        ]
    )


    # ======================================================
    # ÉTAPE 15
    # SAUVEGARDE DU MODÈLE
    # ======================================================

    print("\n")
    print("=" * 70)
    print("ÉTAPE 14 - SAUVEGARDE DU MODÈLE")
    print("=" * 70)


    joblib.dump(

        best_model,

        MODEL_FILE
    )


    print(
        "Modèle sélectionné :",
        best_model_name
    )

    print(
        "Modèle sauvegardé :",
        MODEL_FILE
    )


    # ======================================================
    # ÉTAPE 16
    # SAUVEGARDE DU LABEL ENCODER
    # ======================================================

    joblib.dump(

        label_encoder,

        LABEL_ENCODER_FILE
    )


    print(
        "Label Encoder sauvegardé :",
        LABEL_ENCODER_FILE
    )


    # ======================================================
    # RÉSUMÉ FINAL
    # ======================================================

    print("\n")
    print("=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)

    print(
        f"Nombre de patients : "
        f"{len(df):,}"
    )

    print(
        f"Nombre de variables : "
        f"{X.shape[1]}"
    )

    print(
        f"Modèle retenu : "
        f"{best_model_name}"
    )


    best_row = (
        results_df
        .iloc[0]
    )


    print(
        f"Accuracy : "
        f"{best_row['Accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_row['Precision']:.4f}"
    )

    print(
        f"Recall : "
        f"{best_row['Recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{best_row['F1_Score']:.4f}"
    )

    print(
        f"ROC-AUC : "
        f"{best_row['ROC_AUC']:.4f}"
    )

    print(
        f"CV F1 : "
        f"{best_row['CV_F1_Mean']:.4f}"
    )


    print("\n")
    print("=" * 70)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 70)


    return (
        results_df,
        best_model,
        label_encoder
    )


# ==========================================================
# 6. EXÉCUTION
# ==========================================================

if __name__ == "__main__":

    train_models()

