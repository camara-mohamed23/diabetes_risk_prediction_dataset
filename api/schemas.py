
# ==========================================================
# DIABETES RISK PREDICTION AI
# FASTAPI - SCHEMAS
# ==========================================================

from pydantic import BaseModel, Field


# ==========================================================
# DONNEES PATIENT
# ==========================================================

class PatientData(BaseModel):
    """
    Données envoyées par l'application Streamlit
    vers l'API FastAPI.
    """

    Age: float = Field(..., ge=0, le=120)

    Gender: str
    Country: str

    Height_cm: float = Field(..., gt=0)
    Weight_kg: float = Field(..., gt=0)
    BMI: float = Field(..., gt=0)

    Waist_Circumference_cm: float = Field(..., gt=0)

    Blood_Glucose: float = Field(..., ge=0)
    HbA1c: float = Field(..., ge=0)
    Fasting_Blood_Sugar: float = Field(..., ge=0)
    Insulin_Level: float = Field(..., ge=0)

    Blood_Pressure_Systolic: float = Field(..., ge=0)
    Blood_Pressure_Diastolic: float = Field(..., ge=0)

    Total_Cholesterol: float = Field(..., ge=0)
    HDL: float = Field(..., ge=0)
    LDL: float = Field(..., ge=0)
    Triglycerides: float = Field(..., ge=0)

    Heart_Rate: float = Field(..., ge=0)

    Physical_Activity_Level: str

    Exercise_Hours_Per_Week: float = Field(..., ge=0)
    Daily_Walking_Minutes: float = Field(..., ge=0)

    Diet_Quality: str
    Sugar_Intake_Level: str

    Sleep_Hours: float = Field(..., ge=0, le=24)

    Stress_Level: str

    Smoking_Status: str
    Alcohol_Consumption: str

    Family_History_Diabetes: str

    Hypertension: str
    Heart_Disease: str
    Fatty_Liver: str
    PCOS: str

    Medication_Adherence: str

    Work_Type: str
    Residence_Type: str

    Daily_Water_Intake_L: float = Field(..., ge=0)


# ==========================================================
# REPONSE PREDICTION
# ==========================================================

class PredictionResponse(BaseModel):

    prediction: str

    probabilities: dict[str, float]


# ==========================================================
# HEALTH CHECK
# ==========================================================

class HealthResponse(BaseModel):

    status: str

    model_loaded: bool

    label_encoder_loaded: bool


# ==========================================================
# INFORMATIONS MODELE
# ==========================================================

class ModelInfoResponse(BaseModel):

    model_type: str

    classes: list[str]

    number_of_classes: int

