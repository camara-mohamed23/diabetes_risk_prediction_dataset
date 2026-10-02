# ==========================================================
# DIABETES RISK PREDICTION
# STREAMLIT APPLICATION
# ==========================================================

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="Diabetes Risk Analytics",

    layout="wide"
)


# ==========================================================
# CHARGEMENT
# ==========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "models/diabetes_risk_model.pkl"
    )


@st.cache_data
def load_dataset():

    return pd.read_csv(
        "data/diabetes_risk_prediction_dataset.csv"
    )


model = load_model()

df = load_dataset()


# ==========================================================
# HEADER
# ==========================================================

st.title(
    " Diabetes Risk Analytics"
)

st.markdown(
    """
    ### Data Science & Machine Learning Platform

    Analyse exploratoire et prédiction du niveau
    de risque de diabète.
    """
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(

    "Choisir une section",

    [
        "Dashboard",
        "Data Explorer",
        "Risk Analysis",
        "Patient Prediction"
    ]
)


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.header(
        "Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Patients",
        f"{len(df):,}"
    )

    col2.metric(
        "Variables",
        len(df.columns)
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    col4.metric(
        "Duplicates",
        int(df.duplicated().sum())
    )

    st.divider()

    # ------------------------------------------------------
    # Risk distribution
    # ------------------------------------------------------

    st.subheader(
        "Distribution du risque"
    )

    risk_counts = (
        df["Diabetes_Risk"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Diabetes_Risk",
        "Count"
    ]

    fig = px.bar(

        risk_counts,

        x="Diabetes_Risk",

        y="Count",

        title="Distribution des niveaux de risque"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# DATA EXPLORER
# ==========================================================

elif page == "Data Explorer":

    st.header(
        "Data Explorer"
    )

    st.write(
        "Dimensions :",
        df.shape
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader(
        "Statistiques descriptives"
    )

    st.dataframe(
        df.describe(
            include="all"
        ).T,
        use_container_width=True
    )


# ==========================================================
# RISK ANALYSIS
# ==========================================================

elif page == "Risk Analysis":

    st.header(
        "Risk Analysis"
    )

    numerical_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    selected_variable = st.selectbox(

        "Choisir une variable",

        numerical_columns
    )

    fig = px.box(

        df,

        x="Diabetes_Risk",

        y=selected_variable,

        color="Diabetes_Risk",

        title=(
            f"{selected_variable} "
            "selon le niveau de risque"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------

    st.subheader(
        "Analyse bivariée"
    )

    col1, col2 = st.columns(2)

    with col1:

        x_variable = st.selectbox(
            "Variable X",
            numerical_columns,
            index=0
        )

    with col2:

        y_variable = st.selectbox(
            "Variable Y",
            numerical_columns,
            index=min(
                1,
                len(numerical_columns) - 1
            )
        )

    fig2 = px.scatter(

        df,

        x=x_variable,
        y=y_variable,
        color="Diabetes_Risk",

        hover_data=[
            "Age",
            "Gender"
        ],

        title=(
            f"{x_variable} vs {y_variable}"
        )
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ==========================================================
# PATIENT PREDICTION
# ==========================================================

elif page == "Patient Prediction":

    st.header(
        "Patient Risk Prediction"
    )

    st.info(
        "Cette prédiction est destinée à "
        "l'analyse expérimentale et ne constitue "
        "pas un diagnostic médical."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=40
        )

        gender = st.selectbox(
            "Gender",
            df["Gender"].dropna().unique()
        )

        country = st.selectbox(
            "Country",
            df["Country"].dropna().unique()
        )

        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=175.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=80.0,
            value=26.0
        )

        waist = st.number_input(
            "Waist Circumference (cm)",
            min_value=30.0,
            max_value=200.0,
            value=90.0
        )

        glucose = st.number_input(
            "Blood Glucose",
            min_value=0.0,
            max_value=500.0,
            value=100.0
        )

        hba1c = st.number_input(
            "HbA1c",
            min_value=0.0,
            max_value=20.0,
            value=5.5
        )

        fasting = st.number_input(
            "Fasting Blood Sugar",
            min_value=0.0,
            max_value=500.0,
            value=100.0
        )

        insulin = st.number_input(
            "Insulin Level",
            min_value=0.0,
            max_value=500.0,
            value=20.0
        )

    with col2:

        systolic = st.number_input(
            "Systolic Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

        diastolic = st.number_input(
            "Diastolic Blood Pressure",
            min_value=30,
            max_value=150,
            value=80
        )

        cholesterol = st.number_input(
            "Total Cholesterol",
            min_value=50.0,
            max_value=500.0,
            value=190.0
        )

        hdl = st.number_input(
            "HDL",
            min_value=10.0,
            max_value=150.0,
            value=50.0
        )

        ldl = st.number_input(
            "LDL",
            min_value=10.0,
            max_value=400.0,
            value=120.0
        )

        triglycerides = st.number_input(
            "Triglycerides",
            min_value=20.0,
            max_value=1000.0,
            value=150.0
        )

        heart_rate = st.number_input(
            "Heart Rate",
            min_value=30,
            max_value=220,
            value=75
        )

        physical_activity = st.selectbox(
            "Physical Activity Level",
            df[
                "Physical_Activity_Level"
            ].dropna().unique()
        )

        exercise = st.number_input(
            "Exercise Hours / Week",
            min_value=0.0,
            max_value=50.0,
            value=3.0
        )

        walking = st.number_input(
            "Daily Walking Minutes",
            min_value=0.0,
            max_value=500.0,
            value=30.0
        )

        diet = st.selectbox(
            "Diet Quality",
            df[
                "Diet_Quality"
            ].dropna().unique()
        )

        sugar = st.selectbox(
            "Sugar Intake Level",
            df[
                "Sugar_Intake_Level"
            ].dropna().unique()
        )

    # ======================================================
    # AUTRES VARIABLES
    # ======================================================

    st.subheader(
        "Lifestyle & Medical History"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        sleep = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0
        )

        stress = st.selectbox(
            "Stress Level",
            df[
                "Stress_Level"
            ].dropna().unique()
        )

        smoking = st.selectbox(
            "Smoking Status",
            df[
                "Smoking_Status"
            ].dropna().unique()
        )

    with col2:

        alcohol = st.selectbox(
            "Alcohol Consumption",
            df[
                "Alcohol_Consumption"
            ].dropna().unique()
        )

        family_history = st.selectbox(
            "Family History Diabetes",
            df[
                "Family_History_Diabetes"
            ].dropna().unique()
        )

        hypertension = st.selectbox(
            "Hypertension",
            df[
                "Hypertension"
            ].dropna().unique()
        )

    with col3:

        heart_disease = st.selectbox(
            "Heart Disease",
            df[
                "Heart_Disease"
            ].dropna().unique()
        )

        fatty_liver = st.selectbox(
            "Fatty Liver",
            df[
                "Fatty_Liver"
            ].dropna().unique()
        )

        pcos = st.selectbox(
            "PCOS",
            df[
                "PCOS"
            ].dropna().unique()
        )

    medication = st.selectbox(
        "Medication Adherence",
        df[
            "Medication_Adherence"
        ].dropna().unique()
    )

    work_type = st.selectbox(
        "Work Type",
        df[
            "Work_Type"
        ].dropna().unique()
    )

    residence = st.selectbox(
        "Residence Type",
        df[
            "Residence_Type"
        ].dropna().unique()
    )

    water = st.number_input(
        "Daily Water Intake (L)",
        min_value=0.0,
        max_value=20.0,
        value=2.0
    )

    # ======================================================
    # PREDICTION
    # ======================================================

    if st.button(
        "Predict Diabetes Risk",
        type="primary"
    ):

        patient = {

            "Age": age,

            "Gender": gender,

            "Country": country,

            "Height_cm": height,

            "Weight_kg": weight,

            "BMI": bmi,

            "Waist_Circumference_cm": waist,

            "Blood_Glucose": glucose,

            "HbA1c": hba1c,

            "Fasting_Blood_Sugar": fasting,

            "Insulin_Level": insulin,

            "Blood_Pressure_Systolic": systolic,

            "Blood_Pressure_Diastolic": diastolic,

            "Total_Cholesterol": cholesterol,

            "HDL": hdl,

            "LDL": ldl,

            "Triglycerides": triglycerides,

            "Heart_Rate": heart_rate,

            "Physical_Activity_Level":
                physical_activity,

            "Exercise_Hours_Per_Week":
                exercise,

            "Daily_Walking_Minutes":
                walking,

            "Diet_Quality":
                diet,

            "Sugar_Intake_Level":
                sugar,

            "Sleep_Hours":
                sleep,

            "Stress_Level":
                stress,

            "Smoking_Status":
                smoking,

            "Alcohol_Consumption":
                alcohol,

            "Family_History_Diabetes":
                family_history,

            "Hypertension":
                hypertension,

            "Heart_Disease":
                heart_disease,

            "Fatty_Liver":
                fatty_liver,

            "PCOS":
                pcos,

            "Medication_Adherence":
                medication,

            "Work_Type":
                work_type,

            "Residence_Type":
                residence,

            "Daily_Water_Intake_L":
                water
        }

        patient_df = pd.DataFrame(
            [patient]
        )

        prediction = model.predict(
            patient_df
        )[0]

        st.success(
            f"Résultat du modèle : "
            f"{prediction}"
        )

        # --------------------------------------------------
        # Probabilités
        # --------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    patient_df
                )[0]
            )

            classes = model.classes_

            probability_df = pd.DataFrame({

                "Risk": classes,

                "Probability": (
                    probabilities * 100
                )
            })

            fig = px.bar(

                probability_df,

                x="Risk",

                y="Probability",

                text="Probability",

                title="Probabilité par niveau de risque"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )