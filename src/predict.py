# ==========================================================
# DIABETES RISK PREDICTION
# PREDICTION NOUVEAU PATIENT
# ==========================================================

import joblib
import pandas as pd

from config import MODEL_FILE


# ==========================================================
# CHARGEMENT DU MODÈLE
# ==========================================================

model = joblib.load(
    MODEL_FILE
)


# ==========================================================
# NOUVEAU PATIENT
# ==========================================================

patient = {

    "Age": 45,

    "Gender": "Male",

    "Country": "France",

    "Height_cm": 175,

    "Weight_kg": 85,

    "BMI": 27.8,

    "Waist_Circumference_cm": 96,

    "Blood_Glucose": 120,

    "HbA1c": 6.2,

    "Fasting_Blood_Sugar": 115,

    "Insulin_Level": 20,

    "Blood_Pressure_Systolic": 135,

    "Blood_Pressure_Diastolic": 85,

    "Total_Cholesterol": 200,

    "HDL": 45,

    "LDL": 125,

    "Triglycerides": 160,

    "Heart_Rate": 78,

    "Physical_Activity_Level": "Moderate",

    "Exercise_Hours_Per_Week": 3,

    "Daily_Walking_Minutes": 45,

    "Diet_Quality": "Average",

    "Sugar_Intake_Level": "Moderate",

    "Sleep_Hours": 7,

    "Stress_Level": "Moderate",

    "Smoking_Status": "Never",

    "Alcohol_Consumption": "Never",

    "Family_History_Diabetes": "No",

    "Hypertension": "No",

    "Heart_Disease": "No",

    "Fatty_Liver": "No",

    "PCOS": "No",

    "Medication_Adherence": "Good",

    "Work_Type": "Private",

    "Residence_Type": "Urban",

    "Daily_Water_Intake_L": 2.0
}


# ==========================================================
# DATAFRAME
# ==========================================================

patient_df = pd.DataFrame(
    [patient]
)


# ==========================================================
# PREDICTION
# ==========================================================

prediction = model.predict(
    patient_df
)


print("=" * 70)
print("DIABETES RISK PREDICTION")
print("=" * 70)

print(
    "Risque prédit :",
    prediction[0]
)


# ==========================================================
# PROBABILITÉS
# ==========================================================

if hasattr(
    model,
    "predict_proba"
):

    probabilities = (
        model.predict_proba(
            patient_df
        )[0]
    )

    classes = (
        model.classes_
    )

    print("\nProbabilités :")

    for class_name, probability in zip(
        classes,
        probabilities
    ):

        print(
            f"{class_name} : "
            f"{probability * 100:.2f}%"
        )