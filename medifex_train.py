import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import zipfile

# =========================
# === CONFIGURATION =======
# =========================

# Définition des chemins de fichiers et des colonnes
DATA_PATH = "medifex_data.csv" # Assurez-vous que ce fichier est à la racine de votre dépôt GitHub
MODEL_DIR = "models" # Dossier pour sauvegarder les modèles
MODEL_PATH = os.path.join(MODEL_DIR, "medifex_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

FEATURES = [
    'Age', 'Sexe', 'Tension_arterielle', 'Frequence_cardiaque', 'Cholesterol', 'Glycemie', 'BMI', 'SCFA', 'TMAO',
    'Indole', 'CRP', 'CYP2C19*2 (rs4244285)', 'TPMT*3C (rs1142345)', 'NAT2*6 (rs1799930)', 'SLCO1B1 (rs4149056)',
    'VKORC1 (rs9923231)', 'CYP2D6*4 (rs3892097)', 'Statut_Tabagique', 'Consommation_Alcool', 'Niveau_Activite_Physique'
]
TARGET = "Etat_Sante"

# =========================
# === FONCTIONS UTILITAIRES ===
# =========================

@st.cache_data
def load_data(source):
    """
    Charge les données depuis un fichier CSV.
    Peut prendre un chemin de fichier (str) ou un objet UploadedFile de Streamlit.
    """
    df = pd.DataFrame()

    if isinstance(source, str):
        if not os.path.exists(source):
            st.error(f"Fichier de données introuvable : {source}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(source)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier '{source}' : {e}")
            return pd.DataFrame()
    elif hasattr(source, 'read'):
        try:
            df = pd.read_csv(source)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier téléversé : {e}")
            return pd.DataFrame()
    else:
        st.error("Source de données non reconnue. Veuillez fournir un chemin valide ou un fichier à téléverser.")
        return pd.DataFrame()

    if not df.empty:
        for col in ['Age', 'Tension_arterielle', 'Frequence_cardiaque', 'Cholesterol', 'Glycemie', 'BMI', 'SCFA', 'TMAO', 'Indole', 'CRP']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def train_and_evaluate_model(df, features, target, status_placeholder): # Renommé pour être plus clair
    """
    Entraîne un modèle GradientBoostingClassifier avec GridSearchCV
    et évalue ses performances. Sauvegarde le modèle, scaler et encoder.
    """
    if df.empty or len(df) < 2:
        status_placeholder.error("Données insuffisantes pour l'entraînement du modèle.")
        return None, None, None, None

    status_placeholder.info("Étape 1/5: Nettoyage et vérification des données...")
    initial_rows = len(df)
    cols_to_check = [col for col in features + [target] if col in df.columns]
    df.dropna(subset=cols_to_check, inplace=True)
    rows_after_dropna = len(df)

    if initial_rows > rows_after_dropna:
        status_placeholder.warning(f"Attention : {initial_rows - rows_after_dropna} lignes ont été supprimées car elles contenaient des valeurs manquantes dans les features ou la cible.")
    
    if rows_after_dropna < 2:
        status_placeholder.error("Données insuffisantes pour l'entraînement après la suppression des valeurs manquantes.")
        return None, None, None, None

    X = df[features]
    y = df[target]

    if len(y.unique()) < 2:
        status_placeholder.error(f"La variable cible '{target}' contient moins de 2 classes uniques ({y.unique()}) après le nettoyage. Impossible de réaliser une classification.")
        return None, None, None, None

    status_placeholder.info("Étape 2/5: Encodage des variables...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    st.write(f"Classes encodées pour '{target}' : {label_encoder.classes_}")

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            try:
                temp_encoder = LabelEncoder() 
                X[col] = temp_encoder.fit_transform(X[col])
            except Exception as e:
                status_placeholder.warning(f"Impossible d'encoder la colonne catégorielle '{col}' : {e}. Assurez-vous que ses valeurs sont cohérentes.")
    
    status_placeholder.info("Étape 3/5: Séparation et mise à l'échelle des données...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)
    
    if X_train_scaled_df.isnull().sum().sum() > 0:
        status_placeholder.error(f"Des valeurs NaN subsistent dans les données d'entraînement après mise à l'échelle. Impossible de continuer.")
        st.dataframe(X_train_scaled_df.isnull().sum())
        return None, None, None, None
    if not np.isfinite(X_train_scaled_df).all().all():
        status_placeholder.error(f"Des valeurs infinies subsistent dans les données d'entraînement après mise à l'échelle. Impossible de continuer.")
        return None, None, None, None

    status_placeholder.info("Étape 4/5: Entraînement du modèle avec GridSearchCV (cela peut prendre du temps)...")
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }

    # Utilisation de st.spinner pour une indication visuelle d'une longue opération
    with st.spinner("Entraînement en cours, veuillez patienter..."):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled_df, y_train)

    best_model = grid_search.best_estimator_
    st.write(f"Meilleurs hyperparamètres trouvés : {grid_search.best_params_}")

    status_placeholder.info("Étape 5/5: Évaluation du modèle et sauvegarde...")
    y_pred = best_model.predict(X_test_scaled_df)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    status_placeholder.success("Modèle, scaler et encodeur sauvegardés localement. Entraînement terminé!")
    
    return accuracy, report, best_model, label_encoder.classes_

# =========================
# === INTERFACE STREAMLIT ===
# =========================

st.set_page_config(
    page_title="Médifex AI - Entraînement",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Interface d'Entraînement du Modèle Médifex AI")

st.header("1. Chargement des Données")
uploaded_file = st.file_uploader("Chargez un fichier CSV pour l'entraînement", type=["csv"])

df = pd.DataFrame()
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if not df.empty:
        st.success(f"Fichier '{uploaded_file.name}' chargé avec succès. {len(df)} lignes trouvées.")
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
        st.write("Statistiques descriptives :")
        st.dataframe(df.describe())
else:
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        if not df.empty:
            st.info(f"Fichier de données par défaut '{DATA_PATH}' chargé. {len(df)} lignes.")
            st.dataframe(df.head())
    else:
        st.warning("Veuillez charger un fichier CSV pour commencer l'entraînement.")

st.header("2. Entraîner le Modèle IA")
st.info("L'entraînement inclut la recherche des meilleurs hyperparamètres (GridSearchCV). Cela peut prendre du temps en fonction de la taille de vos données et de la puissance de calcul.")

# Placeholder pour les messages d'état en direct
training_status_placeholder = st.empty()

if not df.empty and st.button("Lancer l'entraînement et l'évaluation"):
    st.write("---")
    st.subheader("Processus d'Entraînement")
    
    # Passe le placeholder à la fonction d'entraînement
    accuracy, report_dict, best_model, target_classes = train_and_evaluate_model(
        df, FEATURES, TARGET, training_status_placeholder
    )

    if best_model is not None:
        st.subheader("Résultats de l'Entraînement")
        st.write(f"**Précision (Accuracy) du meilleur modèle : {accuracy:.2%}**")

        if accuracy >= 0.95:
            st.success("Félicitations ! Le modèle a atteint une précision de 95% ou plus.")
        else:
            st.warning(f"Le modèle a atteint une précision de {accuracy:.2%}. L'objectif de 95% n'est pas atteint.")
            st.info("Pour améliorer la précision :")
            st.write("- **Ajoutez plus de données** d'entraînement.")
            st.write("- **Améliorez la qualité de vos données** (gestion des valeurs manquantes, outliers).")
            st.write("- **Faites de l'ingénierie de fonctionnalités** (créez de nouvelles features à partir des existantes).")
            st.write("- **Essayez d'autres modèles** ou ajustez davantage la grille de paramètres.")

        st.subheader("Rapport de Classification Détaillé")
        st.json(report_dict)
        
        st.write("Matrice de Confusion (sur l'ensemble de test) :")
        X_temp = df[FEATURES]
        y_temp_encoded = LabelEncoder().fit_transform(df[TARGET])
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp_encoded, test_size=0.2, random_state=42)
        
        scaler_temp = StandardScaler()
        X_test_scaled_temp = scaler_temp.fit(X_train_temp).transform(X_test_temp) 
        
        y_pred_cm = best_model.predict(X_test_scaled_temp)
        cm = confusion_matrix(y_test_temp, y_pred_cm, labels=[i for i in range(len(target_classes))])
        
        st.write("Matrice de Confusion (sur l'ensemble de test) :")
        cm_df = pd.DataFrame(cm, index=[f"Vrai: {c}" for c in target_classes], columns=[f"Prédit: {c}" for c in target_classes])
        st.dataframe(cm_df)

    st.write("---")

st.header("3. Télécharger les Modèles Entraînés")
st.info("Une fois l'entraînement terminé, téléchargez les modèles pour les inclure dans votre application Streamlit principale (`medifex_ia.py`).")

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
    zip_file_name = "medifex_model_and_scaler.zip"
    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        zipf.write(MODEL_PATH, os.path.basename(MODEL_PATH))
        zipf.write(SCALER_PATH, os.path.basename(SCALER_PATH))
        zipf.write(ENCODER_PATH, os.path.basename(ENCODER_PATH))

    with open(zip_file_name, "rb") as f:
        st.download_button(
            label="📥 Télécharger les modèles (modèle, scaler, encodeur)",
            data=f.read(),
            file_name=zip_file_name,
            mime="application/zip"
        )
    st.success(f"Les modèles sont prêts à être téléchargés depuis le dossier '{MODEL_DIR}'. N'oubliez pas de les committer sur GitHub si vous voulez que votre application principale (`medifex_ia.py`) les utilise.")
else:
    st.warning("Aucun modèle entraîné à télécharger pour le moment. Lancez l'entraînement ci-dessus.")

st.write("---")
st.info("Développé avec ❤️ par Gemini pour Médifex")
