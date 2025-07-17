import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

# =========================
# === CONFIGURATION =======
# =========================

FEATURES = [
    'Age', 'Sexe', 'Tension_arterielle', 'Frequence_cardiaque', 'Cholesterol', 'Glycemie', 'BMI', 'SCFA', 'TMAO', 
    'Indole', 'CRP', 'CYP2C19*2 (rs4244285)', 'TPMT*3C (rs1142345)', 'NAT2*6 (rs1799930)', 'SLCO1B1 (rs4149056)', 
    'VKORC1 (rs9923231)', 'CYP2D6*4 (rs3892097)', 'Statut_Tabagique', 'Consommation_Alcool', 'Niveau_Activite_Physique'
]
TARGET = "Etat_Sante"

MODEL_PATH = "medifex_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"
DEFAULT_CSV = "100kpatients.csv"

# =========================
# === FONCTIONS ===========
# =========================

def clean_dataframe(df):
    # Conversion des colonnes catégorielles en numériques (si besoin)
    if 'Sexe' in df.columns:
        df['Sexe'] = df['Sexe'].replace({'Homme': 0, 'Femme': 1})
    cat_cols = ['Statut_Tabagique', 'Consommation_Alcool', 'Niveau_Activite_Physique']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    df_clean = df.dropna(subset=FEATURES + [TARGET])
    return df_clean

def auto_train_until_95(df, max_attempts=100, progress_callback=None):
    X = df[FEATURES]
    y = df[TARGET]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    best_accuracy = 0
    best_model = None
    best_report = None
    history = []
    for n_attempts in range(max_attempts):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=n_attempts)
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        history.append(acc)
        if progress_callback:
            progress_callback(acc, n_attempts+1, max_attempts)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_report = classification_report(y_test, y_pred, target_names=encoder.classes_)
        if best_accuracy >= 0.95:
            break
    # Sauvegarde
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(best_model, f"medifex_model_{timestamp}.pkl")
    return best_accuracy, best_report, history

def add_patients(df, patients_list):
    for patient in patients_list:
        df = pd.concat([df, pd.DataFrame([patient])], ignore_index=True)
    df_clean = clean_dataframe(df)
    return df_clean

# =========================
# === INTERFACE STREAMLIT =
# =========================

st.set_page_config(page_title="Medifex AI Training", layout="wide")
st.title("Médifex IA - Entraînement interactif")

st.markdown("""
Cette interface vous permet :
- De déposer ou sélectionner un fichier CSV de patients pour l'entraînement
- D'ajouter des patients manuellement ou par lot
- De lancer l'entraînement jusqu'à 95% de précision ou 100 essais
- De suivre la progression et la performance du modèle
""")

# 1. Dépôt ou sélection du fichier CSV
st.header("1. Chargement du dataset de patients")
uploaded_file = st.file_uploader("Déposez un fichier CSV ou laissez vide pour utiliser le dataset par défaut", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"{len(df)} patients chargés depuis le fichier uploadé.")
else:
    if os.path.exists(DEFAULT_CSV):
        df = pd.read_csv(DEFAULT_CSV)
        st.success(f"{len(df)} patients chargés depuis le fichier par défaut ({DEFAULT_CSV}).")
    else:
        df = pd.DataFrame()
        st.warning("Aucun fichier de patients disponible.")

if not df.empty:
    st.write("Aperçu des données :", df.head())

# 2. Ajout manuel/batch de patients
st.header("2. Ajouter des patients (optionnel)")
add_mode = st.radio("Mode d'ajout :", ["Formulaire manuel", "Batch JSON (copier-coller)"])

new_patients = []

if add_mode == "Formulaire manuel":
    with st.form("add_patient_form"):
        cols = st.columns(3)
        patient = {}
        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                if feature in ['Sexe', 'Statut_Tabagique', 'Consommation_Alcool', 'Niveau_Activite_Physique']:
                    val = st.text_input(feature)
                elif feature in ['Age', 'Tension_arterielle', 'Frequence_cardiaque', 'Cholesterol', 'Glycemie', 'BMI', 'SCFA', 'TMAO', 'Indole', 'CRP']:
                    val = st.number_input(feature, value=0.0)
                else:
                    val = st.number_input(feature, value=0)
                patient[feature] = val
        patient[TARGET] = st.selectbox("Etat_Sante", ["Sain", "Malade"])
        submitted = st.form_submit_button("Ajouter ce patient")
        if submitted:
            new_patients.append(patient)
            st.success("Patient ajouté au batch d'ajout.")
elif add_mode == "Batch JSON (copier-coller)":
    batch_json = st.text_area("Collez ici une liste de patients au format JSON", height=200)
    if batch_json:
        try:
            import json
            patients = json.loads(batch_json)
            if isinstance(patients, list):
                new_patients.extend(patients)
                st.success(f"{len(patients)} patients ajoutés au batch.")
            else:
                st.error("Le format JSON doit être une liste de dictionnaires (patients).")
        except Exception as e:
            st.error(f"Erreur de parsing JSON : {e}")

if new_patients:
    df = add_patients(df, new_patients)
    st.info(f"{len(new_patients)} nouveaux patients seront inclus à l'entraînement.")

# 3. Lancement de l'entraînement
st.header("3. Entraînement IA (GradientBoostingClassifier)")

if st.button("Lancer l'entraînement IA"):
    progress_bar = st.progress(0)
    status = st.empty()
    curve_placeholder = st.empty()

    def progress_callback(acc, attempt, max_attempts):
        progress_bar.progress(acc)
        status.text(f"Essai {attempt}/{max_attempts} - Précision : {acc:.2%}")

    st.info("Entraînement en cours...")

    acc, report, history = auto_train_until_95(df, max_attempts=100, progress_callback=progress_callback)

    # Affichage courbe d'apprentissage
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(history)+1), history, marker="o")
    ax.set_xlabel("Essai")
    ax.set_ylabel("Précision (accuracy)")
    ax.set_title("Courbe d'avancement de l'entraînement")
    ax.grid(True)
    curve_placeholder.pyplot(fig)

    if acc >= 0.95:
        st.success(f"Modèle entraîné avec précision de {acc:.2%} !")
    else:
        st.warning(f"Entraînement terminé, meilleure précision atteinte : {acc:.2%}")

    st.subheader("Rapport de classification du meilleur modèle")
    st.code(report)
    st.info("Le modèle, le scaler et l'encodeur ont été sauvegardés sur le serveur.")

st.header("4. Télécharger les modèles entraînés")
if st.button("Télécharger le modèle (.pkl)"):
    with open(MODEL_PATH, "rb") as f:
        st.download_button("Télécharger le modèle IA", f, file_name="medifex_model.pkl")

if st.button("Télécharger le scaler (.pkl)"):
    with open(SCALER_PATH, "rb") as f:
        st.download_button("Télécharger le scaler", f, file_name="scaler.pkl")

if st.button("Télécharger l'encodeur (.pkl)"):
    with open(ENCODER_PATH, "rb") as f:
        st.download_button("Télécharger l'encodeur", f, file_name="encoder.pkl")