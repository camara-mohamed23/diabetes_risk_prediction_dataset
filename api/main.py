# ==========================================================
# DIABETES RISK PREDICTION AI
# FASTAPI BACKEND
# ==========================================================

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException
)

from api.schemas import (
    PatientData,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)

from api.model_service import (
    load_resources,
    resources_loaded,
    predict_patient,
    get_model_info
)


# ==========================================================
# LIFESPAN
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print()
    print("=" * 70)
    print("DIABETES RISK PREDICTION AI")
    print("FASTAPI BACKEND")
    print("=" * 70)

    try:

        load_resources()

        print()
        print("✓ Modèle ML chargé")
        print("✓ Label Encoder chargé")
        print("✓ API prête")
        print()

    except Exception as error:

        print()
        print(
            "✗ ERREUR DE CHARGEMENT"
        )

        print(
            error
        )

        print()

    print("=" * 70)
    print()

    yield


# ==========================================================
# APPLICATION FASTAPI
# ==========================================================

app = FastAPI(

    title="Diabetes Risk Prediction AI",

    description="""
API REST de prédiction du risque de diabète.

Cette API reçoit les caractéristiques d'un patient,
applique le pipeline Machine Learning entraîné
et retourne une prédiction parmi :

- Low
- Moderate
- High

Cette application est destinée à un projet Data Science
et ne constitue pas un outil de diagnostic médical.
""",

    version="1.0.0",

    lifespan=lifespan
)


# ==========================================================
# ROUTE PRINCIPALE
# ==========================================================

@app.get("/")
def root():

    return {

        "application":
            "Diabetes Risk Prediction AI",

        "version":
            "1.0.0",

        "status":
            "running",

        "api":
            "FastAPI",

        "documentation":
            "/docs"
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    loaded = resources_loaded()

    return {

        "status":
            "healthy"
            if loaded
            else "degraded",

        "model_loaded":
            loaded,

        "label_encoder_loaded":
            loaded
    }


# ==========================================================
# MODEL INFO
# ==========================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse
)
def model_info():

    try:

        return get_model_info()

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ==========================================================
# PREDICTION
# ==========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    patient: PatientData
):

    try:

        result = predict_patient(
            patient.model_dump()
        )

        return result

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                "Erreur pendant "
                f"la prédiction : {error}"
            )
        )

