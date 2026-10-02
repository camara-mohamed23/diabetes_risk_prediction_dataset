# ==========================================================
# DIABETES RISK PREDICTION
# CONFIGURATION DU PROJET
# ==========================================================

from pathlib import Path


# ==========================================================
# CHEMINS DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR = BASE_DIR / "reports"


# ==========================================================
# FICHIER DATASET
# ==========================================================

DATA_FILE = (
    DATA_DIR /
    "diabetes_risk_prediction_dataset.csv"
)


# ==========================================================
# FICHIER DU MODÈLE
# ==========================================================

MODEL_FILE = (
    MODELS_DIR /
    "diabetes_risk_model.pkl"
)


# ==========================================================
# FICHIER DES MÉTRIQUES
# ==========================================================

METRICS_FILE = (
    REPORTS_DIR /
    "model_metrics.csv"
)


# ==========================================================
# VARIABLE CIBLE
# ==========================================================

TARGET = "Diabetes_Risk"


# ==========================================================
# COLONNES À NE PAS UTILISER
# ==========================================================

# Patient_ID n'apporte aucune information prédictive.
#
# Diabetes_Risk_Score est potentiellement directement
# lié à la variable cible.
#
# AI_Health_Recommendation et Doctor_Consultation_Needed
# peuvent également être des informations calculées
# après l'évaluation du risque.
#
# On les exclut donc pour éviter la fuite de données.

LEAKAGE_COLUMNS = [
    "Patient_ID",
    "Diabetes_Risk_Score",
    "AI_Health_Recommendation",
    "Doctor_Consultation_Needed"
]


# ==========================================================
# RANDOM STATE
# ==========================================================

RANDOM_STATE = 42


# ==========================================================
# TEST SIZE
# ==========================================================

TEST_SIZE = 0.20


# ==========================================================
# CROSS VALIDATION
# ==========================================================

CV_FOLDS = 5