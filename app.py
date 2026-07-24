"""
Application Streamlit — Générateur de Budget Personnalisé (v2)
------------------------------------------------------------------
Cette application :
1. Récupère la situation du foyer via un questionnaire (Étape 1).
2. Évalue, de façon optionnelle, un profil d'aversion au risque
   (Prudent / Équilibré / Dynamique) via un mini-QCM (Étape 2).
3. Extrait le texte d'une fiche de paie PDF (pdfplumber).
4. Interroge le modèle Gemini (gemini-2.5-flash, via google-genai) pour
   extraire le salaire net, puis calcule une répartition budgétaire
   inspirée de la règle 50/30/20, adaptée au foyer ET au profil de risque.
5. Génère un rapport Markdown structuré et bienveillant, avec graphiques
   (Étape 3).

Prérequis :
    pip install streamlit pdfplumber google-genai plotly pandas

Clé API :
    Ajoutez votre clé dans .streamlit/secrets.toml :
        GEMINI_API_KEY = "votre_cle_api"
"""

import json
import re

import pandas as pd
import pdfplumber
import plotly.graph_objects as go
import streamlit as st
from google import genai

# ----------------------------------------------------------------------
# Configuration générale
# ----------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"

st.set_page_config(page_title="Mon Budget Personnalisé", page_icon="💶", layout="centered")


# ----------------------------------------------------------------------
# CLIENT GEMINI
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def obtenir_client_gemini():
    """
    Initialise (une seule fois, grâce au cache) le client google-genai à
    partir de la clé API stockée dans st.secrets. Retourne None si la clé
    est absente, pour permettre à l'appelant d'afficher un message clair.
    """
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def appeler_gemini(prompt: str) -> str:
    """
    Envoie un prompt au modèle Gemini (gemini-2.5-flash) et retourne le
    texte généré. Centralise la gestion d'erreurs pour les deux usages
    de l'application (extraction du salaire, rédaction du rapport).
    """
    client = obtenir_client_gemini()
    if client is None:
        st.error(
            "Clé API Gemini introuvable. Ajoutez GEMINI_API_KEY dans "
            "`.streamlit/secrets.toml` pour activer l'IA."
        )
        return ""
    try:
        reponse = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (reponse.text or "").strip()
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Gemini : {e}")
        return ""


# ----------------------------------------------------------------------
# EXTRACTION DU TEXTE DU PDF
# ----------------------------------------------------------------------
def extraire_texte_pdf(fichier_pdf) -> str:
    """
    Lit un fichier PDF (fiche de paie) et retourne l'intégralité du texte
    brut extrait, page par page, via pdfplumber.
    """
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
    """
    Demande à Gemini d'extraire le "Salaire Net à payer" du texte de la
    fiche de paie et de le renvoyer en JSON strict. Retourne un float,
    ou None si l'extraction échoue.
    """
    prompt = f"""Tu es un assistant spécialisé dans la lecture de fiches de paie françaises.
Voici le texte brut extrait d'une fiche de paie :

---
{texte_paie}
---

Ta tâche : trouve le montant du "Salaire Net à payer" (parfois appelé
"Net à payer", "Salaire net", "Net payé").

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou
après, sans balises Markdown, au format exact suivant :
{{"salaire_net": 0000.00}}

Si tu ne trouves aucun montant, réponds avec :
{{"salaire_net": null}}
"""
    reponse = appeler_gemini(prompt)
    if not reponse:
        return None

    # On extrait le premier bloc JSON présent dans la réponse (au cas où
    # le modèle ajoute du texte ou des ``` autour, malgré la consigne).
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
# Chaque option est associée à un score de 1 (très prudent) à 3 (très
# dynamique). Questions inspirées d'un questionnaire patrimonial classique
# (horizon de placement, réaction aux baisses de marché, préférence
# sécurité/rendement, épargne de précaution déjà constituée).
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
    """
    Calcule un score moyen (1 à 3) à partir des réponses au QCM de risque,
    puis en déduit un profil textuel : Prudent, Équilibré ou Dynamique.

    `reponses` est un dict {cle_question: libelle_option_choisie}.
    Retourne (profil, score_moyen). Si aucune réponse n'a été fournie,
    retourne ("Non renseigné", 0.0).
    """
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
# MOTEUR DE CALCUL — Logique 50/30/20 adaptée (foyer + risque)
# ----------------------------------------------------------------------
def calculer_repartition(
    salaire_net: float,
    nb_adultes: int,
    nb_enfants: int,
    handicap: bool,
    montant_dette: float,
    profil_risque: str,
) -> dict:
    """
    Calcule la répartition du salaire net selon une règle 50/30/20 adaptée :

    - Charges fixes (Besoins) : 50% du salaire net.
      -> passe à 60% si handicap/frais médicaux importants OU si le foyer
         compte plus de 3 personnes (adultes + enfants).
    - Dettes / Épargne : 20% du salaire net.
      -> si des dettes existent (montant_dette > 0), ce poste sert en
         priorité au remboursement de la dette plutôt qu'à l'épargne.
      -> le sous-partage Épargne de précaution / Investissement dépend du
         profil de risque : un profil Prudent privilégie le livret
         sécurisé, un profil Dynamique privilégie l'investissement.
    - Loisirs / Plaisir : le reste (le solde après les deux postes ci-dessus).

    Retourne un dictionnaire avec les montants en euros et les taux appliqués.
    """
    nb_personnes = nb_adultes + nb_enfants

    # Taux "Besoins"
    taux_besoins = 0.60 if (handicap or nb_personnes > 3) else 0.50
    taux_dette_epargne = 0.20
    taux_loisirs = 1.0 - taux_besoins - taux_dette_epargne

    montant_besoins = round(salaire_net * taux_besoins, 2)
    montant_dette_epargne = round(salaire_net * taux_dette_epargne, 2)
    montant_loisirs = round(salaire_net * taux_loisirs, 2)

    a_des_dettes = montant_dette > 0

    # Sous-répartition du poste "Dette/Épargne" entre épargne de précaution
    # (livret sécurisé) et investissement, selon le profil de risque.
    # Si le foyer a des dettes, tout ce poste est fléché vers le
    # remboursement de la dette (aucune sous-répartition).
    if a_des_dettes:
        part_precaution = montant_dette_epargne
        part_investissement = 0.0
    else:
        repartitions_risque = {
            "Prudent": (0.80, 0.20),
            "Équilibré": (0.50, 0.50),
            "Dynamique": (0.25, 0.75),
            "Non renseigné": (0.60, 0.40),  # valeur par défaut, prudente
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
# GÉNÉRATION DU RAPPORT MARKDOWN VIA L'IA
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
    """
    Construit un prompt détaillé décrivant la situation du foyer, le
    profil de risque et les montants calculés, puis demande à Gemini de
    rédiger un rapport Markdown structuré (diagnostic, tableau, conseils).
    """
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
- Profil d'aversion au risque (investissement) : {profil_risque} (score {score_risque}/3, "Non renseigné" si le foyer n'a pas répondu au QCM optionnel)

RÉPARTITION BUDGÉTAIRE CALCULÉE :
- Charges fixes / Besoins ({repartition['taux_besoins']*100:.0f}%) : {repartition['montant_besoins']:.2f} €
- Dettes / Épargne ({repartition['taux_dette_epargne']*100:.0f}%) : {repartition['montant_dette_epargne']:.2f} €
{detail_epargne}
- Loisirs / Plaisir ({repartition['taux_loisirs']*100:.0f}%) : {repartition['montant_loisirs']:.2f} €

CONSIGNES DE RÉDACTION :
Structure le rapport en exactement 3 parties, avec des titres Markdown (##) :

1. **Diagnostic rapide du foyer** : un court paragraphe résumant la
   situation (taille du foyer, présence de dettes, contexte médical
   éventuel, profil de risque si renseigné) et le niveau de tension
   budgétaire.

2. **Répartition budgétaire recommandée** : présente un tableau Markdown
   (colonnes : Poste | Pourcentage | Montant en €) reprenant les postes
   ci-dessus (en détaillant la sous-répartition précaution/investissement
   si elle existe), avec une phrase expliquant pourquoi ces pourcentages
   ont été retenus pour ce foyer et ce profil de risque.

3. **3 conseils pratiques et bienveillants** : liste numérotée de 3
   conseils concrets, adaptés à la situation réelle du foyer (dettes,
   handicap, taille du foyer, profil de risque), sans jugement, avec un
   ton encourageant.

Réponds uniquement avec le rapport en Markdown, sans phrase d'introduction
avant le titre.
"""
    return appeler_gemini(prompt)


# ----------------------------------------------------------------------
# INITIALISATION DE L'ÉTAT DE SESSION
# ----------------------------------------------------------------------
if "resultat" not in st.session_state:
    st.session_state.resultat = None  # stockera (repartition, rapport) après génération


# ----------------------------------------------------------------------
# EN-TÊTE
# ----------------------------------------------------------------------
st.title("💶 Mon Budget Personnalisé")
st.caption(
    "Renseignez votre situation, importez votre fiche de paie, et obtenez "
    "une répartition budgétaire générée par IA (Gemini 2.5 Flash)."
)

etape1, etape2, etape3 = st.tabs(
    ["1️⃣ Situation & Documents", "2️⃣ Profil de risque (optionnel)", "3️⃣ Diagnostic & Budget"]
)


# ----------------------------------------------------------------------
# ÉTAPE 1 — SITUATION DU FOYER & DOCUMENTS
# ----------------------------------------------------------------------
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
        mensualite_dette = st.number_input(
            "Mensualité de remboursement (€)", min_value=0.0, value=0.0, step=10.0, format="%.2f"
        )

    st.subheader("📄 Fiche de paie")
    fichier_pdf = st.file_uploader("Importer votre fiche de paie (PDF)", type=["pdf"])
    if fichier_pdf is not None:
        st.success("Fichier importé. Rendez-vous dans l'onglet 3 pour lancer l'analyse.")

    st.info("Passez ensuite à l'onglet **2️⃣ Profil de risque** (facultatif) ou directement à l'onglet **3️⃣**.")


# ----------------------------------------------------------------------
# ÉTAPE 2 — PROFIL DE RISQUE (OPTIONNEL)
# ----------------------------------------------------------------------
with etape2:
    st.subheader("📈 Votre rapport au risque (facultatif)")
    st.caption(
        "Ces questions permettent d'affiner la sous-répartition entre épargne "
        "de précaution et investissement. Vous pouvez laisser ces champs vides "
        "si vous ne souhaitez pas répondre : un profil équilibré par défaut sera utilisé."
    )

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
            st.warning("Répondez à au moins une question pour estimer votre profil.")
    else:
        profil_risque, score_risque = "Non renseigné", 0.0
        st.info("QCM non renseigné : un profil **Équilibré par défaut** sera utilisé pour les recommandations.")


# ----------------------------------------------------------------------
# ÉTAPE 3 — DIAGNOSTIC & BUDGET RECOMMANDÉ
# ----------------------------------------------------------------------
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
                st.error("Le texte n'a pas pu être extrait du PDF (fichier scanné/image ou vide).")
            else:
                with st.expander("Voir le texte extrait de la fiche de paie"):
                    st.text(texte_pdf)

                with st.spinner("Extraction du salaire net via Gemini..."):
                    salaire_net = extraire_salaire_net(texte_pdf)

                if salaire_net is None:
                    st.error(
                        "Le salaire net n'a pas pu être détecté automatiquement. "
                        "Renseignez-le manuellement ci-dessous."
                    )
                    salaire_net = st.number_input(
                        "Salaire net mensuel (€) — saisie manuelle", min_value=0.0, value=0.0, step=10.0
                    )

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

                    with st.spinner("Rédaction du rapport personnalisé via Gemini..."):
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

                    # On mémorise le résultat pour l'affichage (et pour
                    # survivre à un rerun déclenché par un futur widget).
                    st.session_state.resultat = (repartition, rapport)
                else:
                    st.info("Renseignez un salaire net valide pour lancer le calcul du budget.")

    # --- Affichage du résultat le plus récent, s'il existe ---
    if st.session_state.resultat is not None:
        repartition, rapport = st.session_state.resultat

        st.divider()
        st.subheader("📊 Répartition budgétaire")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Besoins", f"{repartition['montant_besoins']:.2f} €", f"{repartition['taux_besoins']*100:.0f}% du net"
        )
        col2.metric(
            "Dettes / Épargne",
            f"{repartition['montant_dette_epargne']:.2f} €",
            f"{repartition['taux_dette_epargne']*100:.0f}% du net",
        )
        col3.metric(
            "Loisirs", f"{repartition['montant_loisirs']:.2f} €", f"{repartition['taux_loisirs']*100:.0f}% du net"
        )

        # Détail de la sous-répartition précaution / investissement,
        # affiché uniquement si le foyer n'a pas de dettes en cours.
        if not repartition["a_des_dettes"]:
            col4, col5 = st.columns(2)
            col4.metric("↳ Épargne de précaution", f"{repartition['part_precaution']:.2f} €")
            col5.metric("↳ Investissement", f"{repartition['part_investissement']:.2f} €")

        # --- Graphique Plotly : répartition en barres ---
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
            st.error("Le rapport n'a pas pu être généré. Vérifiez votre clé API Gemini.")
    elif not lancer:
        st.info("Complétez les onglets 1️⃣ et 2️⃣, puis cliquez sur **Lancer l'analyse IA**.")