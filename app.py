import streamlit as st
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = r"/fin_inclusion_model.pkl"  # place le fichier à la racine du repo ou indique chemin complet

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

st.title("Prédiction - Possession d'un compte bancaire")
st.write("Remplis les champs ci-dessous puis clique sur **Prédire**.")

# Champs d'entrée — adapte les valeurs par défaut/labels si tu veux
country = st.selectbox("Country", options=["Kenya","Uganda","Tanzania","Rwanda","Burundi"])  # adapte si tu veux la liste complète
year = st.number_input("Year", min_value=2000, max_value=2030, value=2018)
location_type = st.selectbox("Location type", options=["Rural","Urban"])
cellphone_access = st.selectbox("Cellphone access", options=["No","Yes"])
household_size = st.number_input("Household size", min_value=1, max_value=50, value=4)
age_of_respondent = st.number_input("Age of respondent", min_value=10, max_value=120, value=30)
gender_of_respondent = st.selectbox("Gender", options=["Male","Female"])
relationship_with_head = st.selectbox("Relationship with head", options=[
    "Head of Household","Spouse","Child","Other"
])
marital_status = st.selectbox("Marital status", options=["Married","Single","Divorced","Widowed"])
education_level = st.selectbox("Education level", options=[
    "No formal education","Primary education","Secondary education","Tertiary education"
])
job_type = st.selectbox("Job type", options=[
    "Self employed","Formally employed Government","Farming and Fishing","Informally employed","Remittance Dependent"
])

# Construire dataframe d'une ligne dans le même format que l'entraînement
input_df = pd.DataFrame([{
    "country": country,
    "year": int(year),
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": int(household_size),
    "age_of_respondent": int(age_of_respondent),
    "gender_of_respondent": gender_of_respondent,
    "relationship_with_head": relationship_with_head,
    "marital_status": marital_status,
    "education_level": education_level,
    "job_type": job_type
}])

if st.button("Prédire"):
    pred = model.predict(input_df)[0]
    # si tu factorizes, 1 correspond à 'Yes' dans mon entraînement ; vérifie selon ton encodage
    st.success("Prédiction : **Yes** — la personne est susceptible d'avoir un compte bancaire." if pred==1 else "Prédiction : **No** — la personne n'est pas susceptible d'avoir un compte bancaire.")
