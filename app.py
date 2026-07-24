"""
Application Streamlit — Générateur de Budget Personnalisé (v2 - Version Groq)
"""

import json
import re

import pandas as pd
import pdfplumber
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------
# Configuration générale
# ----------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Mon Budget Personnalisé", page_icon="💶", layout="centered")


# ----------------------------------------------------------------------
# CLIENT GROQ
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def obtenir_client_groq():
    """
    Initialise le client Groq à partir de la clé API stockée dans st.secrets.
    """
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def appeler_groq(prompt: str) -> str:
    """
    Envoie un prompt au modèle Groq (llama-3.3-70b-versatile) et retourne le texte généré.
    """
    client = obtenir_client_groq()
    if client is None:
        st.error(
            "Clé API Groq introuvable. Ajoutez GROQ_API_KEY dans "
            "les secrets de Streamlit pour activer l'IA."
        )
        return ""
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Groq : {e}")
        return ""


# ----------------------------------------------------------------------
# EXTRACTION DU TEXTE DU PDF
# ----------------------------------------------------------------------
def extraire_texte_pdf(fichier_pdf) -> str:
    texte_complet = ""
    try:
        with pdfplumber.open(fichier_pdf) as pdf:
            for page in pdf.pages:
                texte_page = page.extract_text()
                if texte_page:
                    texte_complet += texte_page + "\n"
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF : {e}")
        return ""
    return texte_complet.strip()


# ----------------------------------------------------------------------
# EXTRACTION DU SALAIRE NET VIA L'IA (format JSON)
# ----------------------------------------------------------------------
def extraire_salaire_net(texte_paie: str) -> float | None:
    prompt = f"""Tu es un assistant spécialisé dans la lecture de fiches de paie françaises.
Voici le texte brut extrait d'une fiche de paie :

---
{texte_paie}
---

Ta tâche : trouve le montant du "Salaire Net à payer" (parfois appelé "Net à payer", "Salaire net", "Net payé").

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou après, sans balises Markdown, au format exact suivant :
{{"salaire_net": 0000.00}}

Si tu ne trouves aucun montant, réponds avec :
{{"salaire_net": null}}
"""
    reponse = appeler_groq(prompt)
    if not reponse:
        return None

    match = re.search(r"\{.*?\}", reponse, re.DOTALL)
    if not match:
        st.warning("L'IA n'a pas renvoyé de JSON exploitable pour le salaire net.")
        return None

    try:
        data = json.loads(match.group(0))
        salaire = data.get("salaire_net")
        return float(salaire) if salaire is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        st.warning("Impossible de parser le JSON renvoyé par l'IA.")
        return None


# ----------------------------------------------------------------------
# PROFIL DE RISQUE — Mini-QCM optionnel
# ----------------------------------------------------------------------
QUESTIONS_RISQUE = {
    "horizon": {
        "label": "Quel est votre horizon de placement pour l'épargne excédentaire ?",
        "options": {
            "Moins de 5 ans": 1,
            "5 à 10 ans": 2,
            "10 à 20 ans": 3,
            "Plus de 20 ans / jusqu'à la retraite": 3,
        },
    },
    "reaction_baisse": {
        "label": "Face à une chute soudaine de 10 à 15% des marchés financiers, quelle est votre réaction ?",
        "options": {
            "Je vends tout immédiatement pour limiter la casse": 1,
            "Je m'inquiète mais je ne touche à rien, j'attends que ça remonte": 2,
            "J'en profite pour investir davantage à prix réduit": 3,
        },
    },
    "preference_repartition": {
        "label": "Si vous deviez choisir la répartition de votre épargne, vous préféreriez :",
        "options": {
            "100% de sécurité (capital garanti, rendement proche de l'inflation)": 1,
            "Un équilibre (une partie sécurisée, une partie investie)": 2,
            "100% de dynamisme (capital non garanti, potentiel de gain élevé)": 3,
        },
    },
    "epargne_precaution": {
        "label": "Quel est le montant de votre épargne de précaution déjà disponible (livrets, compte courant) ?",
        "options": {
            "Moins de 5 000 €": 1,
            "Entre 5 000 € et 15 000 €": 2,
            "Plus de 15 000 €": 3,
        },
    },
}


def calculer_profil_risque(reponses: dict) -> tuple[str, float]:
    scores = []
    for cle, reponse_choisie in reponses.items():
        if reponse_choisie is None:
            continue
        options = QUESTIONS_RISQUE[cle]["options"]
        scores.append(options.get(reponse_choisie, 2))

    if not scores:
        return "Non renseigné", 0.0

    score_moyen = sum(scores) / len(scores)
    if score_moyen <= 1.6:
        profil = "Prudent"
    elif score_moyen <= 2.4:
        profil = "Équilibré"
    else:
        profil = "Dynamique"
    return profil, round(score_moyen, 2)


# ----------------------------------------------------------------------
# MOTEUR DE CALCUL — Logique 50/30/20
# ----------------------------------------------------------------------
def calculer_repartition(
    salaire_net: float,
    nb_adultes: int,
    nb_enfants: int,
    handicap: bool,
    montant_dette: float,
    profil_risque: str,
) -> dict:
    nb_personnes = nb_adultes + nb_enfants

    taux_besoins = 0.60 if (handicap or nb_personnes > 3) else 0.50
    taux_dette_epargne = 0.20
    taux_loisirs = 1.0 - taux_besoins - taux_dette_epargne

    montant_besoins = round(salaire_net * taux_besoins, 2)
    montant_dette_epargne = round(salaire_net * taux_dette_epargne, 2)
    montant_loisirs = round(salaire_net * taux_loisirs, 2)

    a_des_dettes = montant_dette > 0

    if a_des_dettes:
        part_precaution = montant_dette_epargne
        part_investissement = 0.0
    else:
        repartitions_risque = {
            "Prudent": (0.80, 0.20),
            "Équilibré": (0.50, 0.50),
            "Dynamique": (0.25, 0.75),
            "Non renseigné": (0.60, 0.40),
        }
        taux_precaution, taux_investissement = repartitions_risque.get(profil_risque, (0.60, 0.40))
        part_precaution = round(montant_dette_epargne * taux_precaution, 2)
        part_investissement = round(montant_dette_epargne * taux_investissement, 2)

    return {
        "salaire_net": salaire_net,
        "nb_personnes": nb_personnes,
        "taux_besoins": taux_besoins,
        "taux_dette_epargne": taux_dette_epargne,
        "taux_loisirs": taux_loisirs,
        "montant_besoins": montant_besoins,
        "montant_dette_epargne": montant_dette_epargne,
        "montant_loisirs": montant_loisirs,
        "a_des_dettes": a_des_dettes,
        "part_precaution": part_precaution,
        "part_investissement": part_investissement,
    }


# ----------------------------------------------------------------------
# GÉNÉRATION DU RAPPORT MARKDOWN VIA GROQ
# ----------------------------------------------------------------------
def generer_rapport(
    repartition: dict,
    nb_adultes: int,
    nb_enfants: int,
    handicap: bool,
    montant_dette: float,
    mensualite_dette: float,
    profil_risque: str,
    score_risque: float,
) -> str:
    if repartition["a_des_dettes"]:
        detail_epargne = (
            f"- Le poste Dette/Épargne ({repartition['montant_dette_epargne']:.2f} €) est "
            "intégralement fléché vers le remboursement des dettes existantes."
        )
    else:
        detail_epargne = (
            f"- Sous-répartition du poste Dette/Épargne : "
            f"{repartition['part_precaution']:.2f} € en épargne de précaution (livret sécurisé) "
            f"et {repartition['part_investissement']:.2f} € orientés investissement, "
            f"cohérent avec un profil de risque {profil_risque}."
        )

    prompt = f"""Tu es un conseiller budgétaire et patrimonial, bienveillant et pédagogue.
Rédige un rapport en Markdown clair et structuré à partir des données suivantes.

DONNÉES DU FOYER :
- Nombre d'adultes : {nb_adultes}
- Nombre d'enfants : {nb_enfants}
- Handicap / frais médicaux importants : {"Oui" if handicap else "Non"}
- Montant total des dettes : {montant_dette:.2f} €
- Mensualité de remboursement des dettes : {mensualite_dette:.2f} €
- Salaire net mensuel : {repartition['salaire_net']:.2f} €
- Profil d'aversion au risque (investissement) : {profil_risque} (score {score_risque}/3)

RÉPARTITION BUDGÉTAIRE CALCULÉE :
- Charges fixes / Besoins ({repartition['taux_besoins']*100:.0f}%) : {repartition['montant_besoins']:.2f} €
- Dettes / Épargne ({repartition['taux_dette_epargne']*100:.0f}%) : {repartition['montant_dette_epargne']:.2f} €
{detail_epargne}
- Loisirs / Plaisir ({repartition['taux_loisirs']*100:.0f}%) : {repartition['montant_loisirs']:.2f} €

CONSIGNES DE RÉDACTION :
Structure le rapport en exactement 3 parties, avec des titres Markdown (##) :

1. **Diagnostic rapide du foyer** : un court paragraphe résumant la situation et le niveau de tension budgétaire.
2. **Répartition budgétaire recommandée** : présente un tableau Markdown (colonnes : Poste | Pourcentage | Montant en €) reprenant les postes ci-dessus.
3. **3 conseils pratiques et bienveillants** : liste numérotée de 3 conseils concrets et adaptés.

Réponds uniquement avec le rapport en Markdown, sans phrase d'introduction avant le titre.
"""
    return appeler_groq(prompt)


# ----------------------------------------------------------------------
# INITIALISATION ET INTERFACE
# ----------------------------------------------------------------------
if "resultat" not in st.session_state:
    st.session_state.resultat = None

st.title("💶 Mon Budget Personnalisé")
st.caption("Renseignez votre situation, importez votre fiche de paie et obtenez votre bilan IA via Groq.")

etape1, etape2, etape3 = st.tabs(
    ["1️⃣ Situation & Documents", "2️⃣ Profil de risque (optionnel)", "3️⃣ Diagnostic & Budget"]
)

with etape1:
    st.subheader("👨‍👩‍👧 Votre foyer")
    col_a, col_b = st.columns(2)
    with col_a:
        nb_adultes = st.number_input("Nombre d'adultes dans le foyer", min_value=1, max_value=10, value=1, step=1)
    with col_b:
        nb_enfants = st.number_input("Nombre d'enfants dans le foyer", min_value=0, max_value=10, value=0, step=1)

    handicap = st.checkbox("Handicap / Frais médicaux importants")

    st.subheader("💳 Dettes")
    col_c, col_d = st.columns(2)
    with col_c:
        montant_dette = st.number_input("Montant total des dettes (€)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    with col_d:
        mensualite_dette = st.number_input("Mensualité de remboursement (€)", min_value=0.0, value=0.0, step=10.0, format="%.2f")

    st.subheader("📄 Fiche de paie")
    fichier_pdf = st.file_uploader("Importer votre fiche de paie (PDF)", type=["pdf"])
    if fichier_pdf is not None:
        st.success("Fichier importé. Rendez-vous dans l'onglet 3 pour lancer l'analyse.")

with etape2:
    st.subheader("📈 Votre rapport au risque (facultatif)")
    repondre_qcm = st.checkbox("Je souhaite renseigner mon profil de risque")
    reponses_risque = {"horizon": None, "reaction_baisse": None, "preference_repartition": None, "epargne_precaution": None}

    if repondre_qcm:
        for cle, question in QUESTIONS_RISQUE.items():
            reponses_risque[cle] = st.radio(
                question["label"],
                options=list(question["options"].keys()),
                index=None,
                key=f"risque_{cle}",
            )
        profil_risque, score_risque = calculer_profil_risque(reponses_risque)
        if profil_risque != "Non renseigné":
            st.success(f"Profil de risque estimé : **{profil_risque}** (score {score_risque}/3)")
    else:
        profil_risque, score_risque = "Non renseigné", 0.0
        st.info("Profil Équilibré par défaut retenu.")

with etape3:
    st.subheader("🚀 Générer mon diagnostic budgétaire")
    lancer = st.button("Lancer l'analyse IA", use_container_width=True, type="primary")

    if lancer:
        if fichier_pdf is None:
            st.warning("Merci d'importer votre fiche de paie (onglet 1️⃣) avant de continuer.")
        else:
            with st.spinner("Lecture du PDF en cours..."):
                texte_pdf = extraire_texte_pdf(fichier_pdf)

            if not texte_pdf:
                st.error("Le texte n'a pas pu être extrait du PDF.")
            else:
                with st.spinner("Extraction du salaire net via Groq..."):
                    salaire_net = extraire_salaire_net(texte_pdf)

                if salaire_net is None:
                    st.error("Le salaire net n'a pas pu être détecté automatiquement. Renseignez-le ci-dessous.")
                    salaire_net = st.number_input("Salaire net mensuel (€) — saisie manuelle", min_value=0.0, value=0.0, step=10.0)

                if salaire_net and salaire_net > 0:
                    st.success(f"Salaire net détecté : **{salaire_net:.2f} €**")

                    repartition = calculer_repartition(
                        salaire_net=salaire_net,
                        nb_adultes=nb_adultes,
                        nb_enfants=nb_enfants,
                        handicap=handicap,
                        montant_dette=montant_dette,
                        profil_risque=profil_risque,
                    )

                    with st.spinner("Rédaction du rapport personnalisé via Groq..."):
                        rapport = generer_rapport(
                            repartition=repartition,
                            nb_adultes=nb_adultes,
                            nb_enfants=nb_enfants,
                            handicap=handicap,
                            montant_dette=montant_dette,
                            mensualite_dette=mensualite_dette,
                            profil_risque=profil_risque,
                            score_risque=score_risque,
                        )

                    st.session_state.resultat = (repartition, rapport)

    if st.session_state.resultat is not None:
        repartition, rapport = st.session_state.resultat

        st.divider()
        st.subheader("📊 Répartition budgétaire")

        col1, col2, col3 = st.columns(3)
        col1.metric("Besoins", f"{repartition['montant_besoins']:.2f} €", f"{repartition['taux_besoins']*100:.0f}% du net")
        col2.metric("Dettes / Épargne", f"{repartition['montant_dette_epargne']:.2f} €", f"{repartition['taux_dette_epargne']*100:.0f}% du net")
        col3.metric("Loisirs", f"{repartition['montant_loisirs']:.2f} €", f"{repartition['taux_loisirs']*100:.0f}% du net")

        if not repartition["a_des_dettes"]:
            col4, col5 = st.columns(2)
            col4.metric("↳ Épargne de précaution", f"{repartition['part_precaution']:.2f} €")
            col5.metric("↳ Investissement", f"{repartition['part_investissement']:.2f} €")

        df_repartition = pd.DataFrame(
            {
                "Poste": ["Besoins", "Dettes / Épargne", "Loisirs"],
                "Montant (€)": [
                    repartition["montant_besoins"],
                    repartition["montant_dette_epargne"],
                    repartition["montant_loisirs"],
                ],
            }
        )
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_repartition["Poste"],
                    y=df_repartition["Montant (€)"],
                    text=df_repartition["Montant (€)"].map(lambda v: f"{v:.0f} €"),
                    textposition="outside",
                    marker_color=["#4C78A8", "#F58518", "#54A24B"],
                )
            ]
        )
        fig.update_layout(
            title="Répartition mensuelle du salaire net",
            yaxis_title="Montant (€)",
            showlegend=False,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📝 Rapport détaillé")
        if rapport:
            st.markdown(rapport)
        else:
            st.error("Le rapport n'a pas pu être généré. Vérifiez votre clé API Groq.")
