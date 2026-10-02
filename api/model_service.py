
# ==========================================================
# DIABETES RISK PREDICTION AI
# FASTAPI - MODEL SERVICE
# ==========================================================

from pathlib import Path

import joblib
import pandas as pd


# ==========================================================
# CHEMIN PRINCIPAL DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# FICHIERS MODELES
# ==========================================================

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "diabetes_risk_model.pkl"
)

LABEL_ENCODER_FILE = (
    BASE_DIR
    / "models"
    / "label_encoder.pkl"
)


# ==========================================================
# VARIABLES GLOBALES
# ==========================================================

model = None

label_encoder = None


# ==========================================================
# CHARGER LE MODELE
# ==========================================================

def load_model():

    global model

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"✓ Modèle chargé : {MODEL_FILE}"
    )

    return model


# ==========================================================
# CHARGER LE LABEL ENCODER
# ==========================================================

def load_label_encoder():

    global label_encoder

    if not LABEL_ENCODER_FILE.exists():

        raise FileNotFoundError(
            "Label encoder introuvable : "
            f"{LABEL_ENCODER_FILE}"
        )

    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    print(
        f"✓ Label encoder chargé : "
        f"{LABEL_ENCODER_FILE}"
    )

    return label_encoder


# ==========================================================
# CHARGER TOUTES LES RESSOURCES
# ==========================================================

def load_resources():

    load_model()

    load_label_encoder()


# ==========================================================
# VERIFIER SI LES RESSOURCES SONT CHARGEES
# ==========================================================

def resources_loaded():

    return (
        model is not None
        and label_encoder is not None
    )


# ==========================================================
# OBTENIR LE TYPE DU MODELE
# ==========================================================

def get_model_type():

    if model is None:
        return "Unknown"

    # Si le modèle est un Pipeline sklearn
    if hasattr(model, "named_steps"):

        if "model" in model.named_steps:

            return type(
                model.named_steps["model"]
            ).__name__

        # Au cas où le nom du dernier step
        # est différent
        last_step = list(
            model.named_steps.values()
        )[-1]

        return type(
            last_step
        ).__name__

    return type(model).__name__


# ==========================================================
# INFORMATIONS DU MODELE
# ==========================================================

def get_model_info():

    if not resources_loaded():

        load_resources()

    return {

        "model_type":
            get_model_type(),

        "classes":
            label_encoder.classes_.tolist(),

        "number_of_classes":
            len(
                label_encoder.classes_
            )
    }


# ==========================================================
# PREDICTION
# ==========================================================

def predict_patient(
    patient_data: dict
):

    if not resources_loaded():

        load_resources()


    # ------------------------------------------------------
    # DICTIONNAIRE → DATAFRAME
    # ------------------------------------------------------

    patient_df = pd.DataFrame(
        [patient_data]
    )


    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    prediction_encoded = model.predict(
        patient_df
    )


    # ------------------------------------------------------
    # CONVERSION DE LA CLASSE
    # ------------------------------------------------------

    prediction_label = (
        label_encoder
        .inverse_transform(
            prediction_encoded
        )[0]
    )


    # ------------------------------------------------------
    # PROBABILITES
    # ------------------------------------------------------

    probabilities = {}


    if hasattr(
        model,
        "predict_proba"
    ):

        probability_values = (
            model.predict_proba(
                patient_df
            )[0]
        )


        # Classes connues par le modèle
        if hasattr(
            model,
            "classes_"
        ):

            model_classes = model.classes_

        else:

            model_classes = range(
                len(
                    probability_values
                )
            )


        for encoded_class, probability in zip(
            model_classes,
            probability_values
        ):

            # Conversion vers la classe texte
            original_class = (
                label_encoder
                .inverse_transform(
                    [int(encoded_class)]
                )[0]
            )

            probabilities[
                original_class
            ] = round(
                float(probability),
                4
            )


    # ------------------------------------------------------
    # RESULTAT
    # ------------------------------------------------------

    return {

        "prediction":
            prediction_label,

        "probabilities":
            probabilities
    }
