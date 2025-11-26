# ===========================
#  TRAINING MODEL SCRIPT
# ===========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ===========================
# 1️⃣ Chargement des données
# ===========================

df = pd.read_csv("Data/Financial_inclusion_dataset.csv")   # ← mettre le bon chemin si nécessaire
print("\n✔ Dataset chargé avec succès !")

# ===========================
# 2️⃣ Préparation des données
# ===========================

# La variable cible → possession de compte bancaire
df["bank_account"] = df["bank_account"].map({"Yes":1, "No":0})

X = df.drop(columns=["bank_account", "uniqueid"])
y = df["bank_account"]

# Séparation colonnes numériques & catégorielles
num_features = ['year', 'household_size', 'age_of_respondent']
cat_features = [col for col in X.columns if col not in num_features]

# ===========================
# 3️⃣ Pipeline de préprocessing + modèle ML
# ===========================

preprocess = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("numeric", "passthrough", num_features)
])

model = Pipeline([
    ("transform", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=250, random_state=42))
])

# ===========================
# 4️⃣ Split et entraînement
# ===========================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# ===========================
# 5️⃣ Évaluation
# ===========================

print("\n🔍 Performance du modèle :")
print("Accuracy :", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# ===========================
# 6️⃣ Sauvegarde du modèle
# ===========================

joblib.dump(model, "fin_inclusion_model.pkl")
print("\n💾 Modèle sauvegardé sous : fin_inclusion_model.pkl")
print("\n🎯 Ton modèle est prêt pour Streamlit !")
