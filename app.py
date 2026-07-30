"""
Aurelion Wealth Management — Diagnostic Budgétaire & Plan d'Investissement
---------------------------------------------------------------------------
Cette application :

1. Recueille l'identité, le foyer, le statut professionnel (salarié,
   fonctionnaire, retraité, demandeur d'emploi, entrepreneur/indépendant
   avec sous-statut), le code postal (pour estimer la tension du coût de
   la vie locale), le logement et le bilan patrimonial complet de l'utilisateur
   (Étape 1). Le salaire net peut être renseigné soit via l'import d'une
   fiche de paie PDF (10 Mo max), soit par saisie manuelle directe.
2. Évalue, de façon optionnelle, un profil d'aversion au risque
   (Prudent / Équilibré / Dynamique) via un mini-QCM resserré à 3
   questions non redondantes, complété d'une question de connaissances
   financières utilisée pour adapter le ton du rapport (Étape 2).
3. Si une fiche de paie a été importée, en extrait le texte (pdfplumber)
   puis en déduit le salaire net via l'IA (Groq / llama-3.3-70b-versatile).
4. Calcule, en deux temps :
     - Temps 1 : Reste à Vivre Réel = Revenu - (Charges fixes + Impôts).
       La catégorie "Plaisirs" est plafonnée à un pourcentage raisonnable
       du revenu (15 à 25%, ajusté selon la zone du code postal et le foyer) : tout
       l'excédent au-delà de ce plafond bascule automatiquement dans la
       capacité d'épargne, au lieu d'être surdimensionné.
     - Temps 2 : l'allocation de la capacité d'épargne (épargne de
       précaution, PER, investissement long terme), selon l'âge, la TMI,
       le statut professionnel, le patrimoine existant et le profil de
       risque.
5. Demande à Groq de rédiger un rapport Markdown concis suivant
   strictement ce plan en deux temps, avec un plan d'investissement
   structuré nommant des supports concrets (Étape 3).

Prérequis :
    pip install streamlit pdfplumber groq plotly pandas pillow

Clé API :
    Ajoutez votre clé dans .streamlit/secrets.toml :
        GROQ_API_KEY = "votre_cle_api"

Limite d'upload :
    La taille maximale des fichiers importés (10 Mo) est fixée au niveau
    serveur via .streamlit/config.toml ([server] maxUploadSize = 10) et
    revérifiée manuellement dans le code par sécurité.

Identité visuelle :
    Le logo doit être placé à côté de ce fichier, dans assets/aurelion_logo.jpg.
    Les couleurs de la charte (bleu marine / or) sont définies dans
    .streamlit/config.toml ([theme]).
"""

import json
import re
import os

import pandas as pd
import pdfplumber
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------
# Configuration générale
# ----------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
TAILLE_MAX_FICHIER_MO = 10
TAILLE_MAX_FICHIER_OCTETS = TAILLE_MAX_FICHIER_MO * 1024 * 1024
CHEMIN_LOGO = "aurelion_logo.jpg"

# Départements français dont le coût de la vie / logement est jugé très élevé :
# Île-de-France (75, 77, 78, 91, 92, 93, 94, 95), PACA littoral (06, 13, 83), Corse (20, 2A, 2B), Genevois français (74)
DEPARTEMENTS_COUT_ELEVE = {"75", "77", "78", "91", "92", "93", "94", "95", "06", "13", "83", "20", "2A", "2B", "74"}

TEXTE_AIDE_TMI = (
    "La Tranche Marginale d'Imposition (TMI) est le taux appliqué à votre dernière tranche "
    "de revenus, et non le pourcentage global payé sur vos revenus (Taux Moyen). "
    "Elle est essentielle pour calculer l'avantage fiscal d'un PER."
)

st.set_page_config(page_title="Aurelion Wealth Management", page_icon=CHEMIN_LOGO if os.path.exists(CHEMIN_LOGO) else "🪙", layout="centered")

# ----------------------------------------------------------------------
# IDENTITÉ VISUELLE — Charte noir / or, style minimaliste et sobre
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    h1, h2, h3 { color: #D4AF6A; font-weight: 600; letter-spacing: 0.3px; }
    hr { border: none; border-top: 1px solid #D4AF6A; margin: 1.4em 0; }
    div.stButton > button[kind="primary"] {
        background-color: #D4AF6A; color: #0B0B0B; border: none; font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover { background-color: #B8943F; color: #000000; }
    div[data-testid="stMetricValue"] { color: #D4AF6A; }
    div[data-testid="stMetricLabel"] { color: #EDE6D3; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #D4AF6A; }
    .stTabs [aria-selected="true"] { color: #D4AF6A !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# CLIENT GROQ
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def obtenir_client_groq():
    """
    Initialise (une seule fois, grâce au cache) le client Groq à partir de
    la clé API stockée dans st.secrets.
    """
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def appeler_groq(prompt: str) -> str:
    """
    Envoie un prompt au modèle Groq (llama-3.3-70b-versatile) et retourne
    le texte généré.
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
    """
    Lit un fichier PDF (fiche de paie) et retourne l'intégralité du texte
    brut extrait via pdfplumber.
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
    Demande à Groq d'extraire le "Salaire Net à payer" du texte de la
    fiche de paie et de le renvoyer en JSON strict.
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
}

CONNAISSANCES_FINANCIERES = {
    "Novice : je n'y connais pas grand-chose, j'ai besoin d'être guidé de A à Z": "Novice",
    "Intermédiaire : je comprends les bases mais je manque de stratégie": "Intermédiaire",
    "Expert : je gère déjà activement mes investissements": "Expert",
}


def calculer_profil_risque(reponses: dict) -> tuple[str, float]:
    """
    Calcule un score moyen (1 à 3) à partir des réponses aux 3 questions
    de risque, puis en déduit un profil textuel.
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
# MOTEUR DE CALCUL — TEMPS 1 : Reste à Vivre Réel & répartition
# ----------------------------------------------------------------------
def determiner_taux_plafond_plaisir_base(revenu_total: float) -> float:
    """
    Barème progressif du plafond "Plaisirs", exprimé en % du revenu total.
    """
    if revenu_total < 2000:
        return 0.22
    elif revenu_total < 3000:
        return 0.20
    elif revenu_total < 4500:
        return 0.17
    elif revenu_total < 6000:
        return 0.14
    else:
        return 0.11


def calculer_budget_temps1(
    salaire_net: float,
    autres_revenus: float,
    montant_logement: float,
    autres_charges_essentielles: float,
    mensualite_dette_hors_immo: float,
    tmi_pct: float,
    handicap: bool,
    nb_personnes: int,
    zone_cout_vie: str,
) -> dict:
    """
    TEMPS 1 : calcule le Reste à Vivre Réel et ventile le budget.
    """
    revenu_total = round(salaire_net + autres_revenus, 2)
    depenses_essentielles = round(montant_logement + autres_charges_essentielles + mensualite_dette_hors_immo, 2)
    impots_mensuels = round(revenu_total * tmi_pct / 100, 2)

    reste_a_vivre_brut = round(revenu_total - depenses_essentielles - impots_mensuels, 2)
    deficit = reste_a_vivre_brut < 0
    reste_a_vivre = max(reste_a_vivre_brut, 0.0)

    taux_plafond_plaisir = determiner_taux_plafond_plaisir_base(revenu_total)
    if zone_cout_vie == "Élevé":
        taux_plafond_plaisir += 0.05
    if handicap:
        taux_plafond_plaisir -= 0.05
    elif nb_personnes > 3:
        taux_plafond_plaisir -= 0.03
    taux_plafond_plaisir = max(taux_plafond_plaisir, 0.08)

    plafond_plaisir_montant = round(revenu_total * taux_plafond_plaisir, 2)
    plaisirs = round(min(reste_a_vivre, plafond_plaisir_montant), 2)
    capacite_epargne = round(reste_a_vivre - plaisirs, 2)

    return {
        "revenu_total": revenu_total,
        "depenses_essentielles": depenses_essentielles,
        "impots_mensuels": impots_mensuels,
        "reste_a_vivre_brut": reste_a_vivre_brut,
        "deficit": deficit,
        "taux_plafond_plaisir": taux_plafond_plaisir,
        "plaisirs": plaisirs,
        "capacite_epargne": capacite_epargne,
    }


# ----------------------------------------------------------------------
# MOTEUR DE CALCUL — TEMPS 2 : Allocation de la capacité d'épargne
# ----------------------------------------------------------------------
SEUIL_MINIMUM_LIGNE_EUROS = 50.0


def appliquer_seuil_minimum(montant: float, seuil: float = SEUIL_MINIMUM_LIGNE_EUROS) -> float:
    return 0.0 if 0 < montant < seuil else round(montant, 2)


def obtenir_bonus_stabilite(statut_professionnel: str, sous_statut: str | None, duree_restante_mois: float | None) -> float:
    mapping = {
        ("Salarié", "CDI"): 0,
        ("Salarié", "CDD"): 2,
        ("Salarié", "Intérim"): 2,
        ("Salarié", "Stage"): 2.5,
        ("Salarié", "Alternance / Apprentissage"): 1.5,
        ("Fonctionnaire", None): -1,
        ("Retraité", None): -1,
        ("Demandeur d'emploi", None): 3,
        ("Entrepreneur / Indépendant", "Auto-entrepreneur / Micro-entreprise"): 2.5,
        ("Entrepreneur / Indépendant", "EI / EIRL"): 2.5,
        ("Entrepreneur / Indépendant", "Société (SASU, EURL, SARL, SAS)"): 2,
        ("Entrepreneur / Indépendant", "Profession Libérale / TNS"): 2,
    }
    bonus = mapping.get((statut_professionnel, sous_statut), 0)

    contrats_courts = {"CDD", "Intérim", "Stage", "Alternance / Apprentissage"}
    if statut_professionnel == "Salarié" and sous_statut in contrats_courts and duree_restante_mois is not None and duree_restante_mois <= 6:
        bonus += 1
    return bonus


def construire_allocation_temps2(
    capacite_epargne: float,
    depenses_essentielles: float,
    epargne_disponible: float,
    age: int,
    tmi_pct: float,
    profil_risque: str,
    statut_professionnel: str,
    sous_statut: str | None,
    duree_restante_mois: float | None,
) -> dict:
    if depenses_essentielles > 0:
        mois_couverture_actuelle = round(epargne_disponible / depenses_essentielles, 1)
    else:
        mois_couverture_actuelle = 99.0

    base_cible = {"Prudent": 6, "Équilibré": 4.5, "Dynamique": 3, "Non renseigné": 4.5}.get(profil_risque, 4.5)
    bonus_stabilite = obtenir_bonus_stabilite(statut_professionnel, sous_statut, duree_restante_mois)
    cible_matelas_mois = min(max(base_cible + bonus_stabilite, 3), 9)

    matelas_rempli = mois_couverture_actuelle >= cible_matelas_mois

    part_dynamique_pct = {"Prudent": 30, "Équilibré": 55, "Dynamique": 80, "Non renseigné": 50}.get(profil_risque, 50)
    if age >= 50:
        part_dynamique_pct -= 10
    elif age < 30:
        part_dynamique_pct += 5
    part_dynamique_pct = min(max(part_dynamique_pct, 10), 90)

    if capacite_epargne <= 0:
        return {
            "mois_couverture_actuelle": mois_couverture_actuelle,
            "cible_matelas_mois": cible_matelas_mois,
            "matelas_rempli": matelas_rempli,
            "montant_matelas": 0.0,
            "montant_per": 0.0,
            "montant_invest_long_terme": 0.0,
            "part_dynamique_pct": part_dynamique_pct,
        }

    if matelas_rempli:
        taux_matelas = 0.0
    elif mois_couverture_actuelle < 3:
        taux_matelas = 0.60
    else:
        taux_matelas = 0.30

    montant_matelas_brut = round(capacite_epargne * taux_matelas, 2)
    montant_matelas = appliquer_seuil_minimum(montant_matelas_brut)
    reste_apres_matelas = round(capacite_epargne - montant_matelas, 2)

    if not matelas_rempli:
        taux_per = 0.0
    elif statut_professionnel == "Retraité":
        taux_per = 0.0
    elif tmi_pct >= 30 and age < 60:
        taux_per = 0.30 if tmi_pct >= 41 else 0.20
    else:
        taux_per = 0.0

    montant_per_brut = round(reste_apres_matelas * taux_per, 2)
    montant_per = appliquer_seuil_minimum(montant_per_brut)

    montant_invest_long_terme = round(reste_apres_matelas - montant_per, 2)

    return {
        "mois_couverture_actuelle": mois_couverture_actuelle,
        "cible_matelas_mois": cible_matelas_mois,
        "matelas_rempli": matelas_rempli,
        "montant_matelas": montant_matelas,
        "montant_per": montant_per,
        "montant_invest_long_terme": montant_invest_long_terme,
        "part_dynamique_pct": part_dynamique_pct,
    }


# ----------------------------------------------------------------------
# GÉNÉRATION DU RAPPORT MARKDOWN VIA GROQ
# ----------------------------------------------------------------------
def generer_rapport(
    identite: dict,
    emploi: dict,
    localisation: dict,
    logement: dict,
    dettes: dict,
    patrimoine: dict,
    temps1: dict,
    temps2: dict,
    profil_risque: str,
    score_risque: float,
    niveau_connaissance: str,
) -> str:
    if emploi["sous_statut"]:
        detail_emploi = f"{emploi['statut_professionnel']} — {emploi['sous_statut']}"
    else:
        detail_emploi = emploi["statut_professionnel"]
    if emploi["duree_restante_mois"] is not None:
        detail_emploi += f" ({emploi['duree_restante_mois']:.0f} mois restants)"

    localisation_txt = f"Code postal {localisation['code_postal']}"

    prompt = f"""Tu es un conseiller en gestion de patrimoine, direct et rigoureux. Ton style est concis : pas de paragraphes longs, va à l'essentiel.
Adapte ton niveau de vocabulaire au niveau de connaissances financières de l'utilisateur : {niveau_connaissance}.
Adapte tes conseils retraite/prévoyance et ta gestion de la trésorerie selon le statut professionnel.

Respecte STRICTEMENT les règles d'allocation suivantes, sans jamais les contredire :
- N'invente jamais une ligne d'allocation mensuelle inférieure à 50 € (sauf 0 €).
- Si l'épargne de précaution affichée est à 0 €, explique que la cible de mois de couverture est déjà atteinte.
- Le PER n'est proposé que si le matelas de sécurité est rempli ET la TMI ≥ 30%.

Rédige un rapport en Markdown à partir des données suivantes.
Commence par une phrase précisant que ce diagnostic est indicatif et ne remplace pas l'avis d'un conseiller en gestion de patrimoine agréé.

IDENTITÉ & FOYER :
- Âge : {identite['age']} ans
- Situation matrimoniale : {identite['situation_matrimoniale']}
- Nombre de parts fiscales : {identite['nb_parts_fiscales']}
- Nombre d'adultes : {identite['nb_adultes']} / Nombre d'enfants : {identite['nb_enfants']}
- Handicap / frais médicaux importants : {"Oui" if identite['handicap'] else "Non"}

STATUT PROFESSIONNEL :
- {detail_emploi}

LOCALISATION :
- {localisation_txt} — zone de coût de la vie estimée : {localisation['zone_cout_vie']}

LOGEMENT :
- Statut : {logement['statut']}
- Montant mensuel logement (loyer ou mensualité crédit) : {logement['montant']:.2f} €
- Autres charges essentielles mensuelles : {logement['autres_charges']:.2f} €

DETTES HORS IMMOBILIER :
- Montant total restant dû : {dettes['montant_total']:.2f} €
- Mensualité de remboursement : {dettes['mensualite']:.2f} €

BILAN PATRIMONIAL :
- Salaire net mensuel : {patrimoine['salaire_net']:.2f} €
- Autres revenus mensuels : {patrimoine['autres_revenus']:.2f} €
- Épargne disponible : {patrimoine['epargne_disponible']:.2f} €
- Épargne long terme existante : {patrimoine['epargne_long_terme']:.2f} €
- Produits déjà détenus : {patrimoine['produits_detenus']}
- Tranche Marginale d'Imposition (TMI) déclarée : {patrimoine['tmi_pct']:.0f}%
- Profil d'aversion au risque : {profil_risque} (score {score_risque}/3)

RÉSULTATS DU MOTEUR DE CALCUL — TEMPS 1 (Reste à Vivre Réel & répartition) :
- Revenu total mensuel : {temps1['revenu_total']:.2f} €
- Dépenses essentielles : {temps1['depenses_essentielles']:.2f} €
- Impôts & taxes estimés : {temps1['impots_mensuels']:.2f} €
- Plaisirs (plafonnés à {temps1['taux_plafond_plaisir']*100:.0f}% du revenu total) : {temps1['plaisirs']:.2f} €
- Capacité d'épargne mensuelle calculée : {temps1['capacite_epargne']:.2f} €
- Situation de déficit budgétaire ce mois-ci : {"Oui" if temps1['deficit'] else "Non"}

RÉSULTATS DU MOTEUR DE CALCUL — TEMPS 2 (Allocation de la capacité d'épargne) :
- Couverture actuelle du matelas de sécurité : {temps2['mois_couverture_actuelle']:.1f} mois de charges essentielles (cible : {temps2['cible_matelas_mois']} mois)
- Matelas de sécurité déjà rempli : {"Oui" if temps2['matelas_rempli'] else "Non"}
- Allocation Épargne de précaution : {temps2['montant_matelas']:.2f} €
- Allocation PER : {temps2['montant_per']:.2f} €
- Allocation Investissement long terme : {temps2['montant_invest_long_terme']:.2f} € (dont environ {temps2['part_dynamique_pct']:.0f}% à orienter vers des supports dynamiques)

CONSIGNES DE RÉDACTION :

## Temps 1 — Reste à vivre réel et répartition du budget mensuel
Diagnostic en 2-3 phrases MAXIMUM.
Tableau Markdown (Poste | Montant en € | % du revenu total) reprenant les 4 postes.
2 à 3 puces courtes expliquant POURQUOI cette répartition.

## Temps 2 — Plan d'investissement personnalisé
Tableau Markdown (Allocation | Montant en € | % de la capacité d'épargne) reprenant les 3 lignes du Temps 2.
Un court paragraphe "Contexte de marché" (3-4 phrases maximum).
Section "Dans quoi investir concrètement" : pour chaque ligne du Temps 2, nomme 1 à 2 supports précis.
"Feuille de route" : liste numérotée de 3 à 5 actions concrètes à réaliser dès ce mois-ci.

Réponds uniquement avec le rapport en Markdown.
"""
    return appeler_groq(prompt)


# ----------------------------------------------------------------------
# INITIALISATION DE L'ÉTAT DE SESSION
# ----------------------------------------------------------------------
if "resultat" not in st.session_state:
    st.session_state.resultat = None


# ----------------------------------------------------------------------
# EN-TÊTE
# ----------------------------------------------------------------------
col_logo, col_titre = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    if os.path.exists(CHEMIN_LOGO):
        st.image(CHEMIN_LOGO, width=88)
with col_titre:
    st.markdown("<h1 style='margin-bottom:0;'>Aurelion Wealth Management</h1>", unsafe_allow_html=True)
    st.caption("Diagnostic budgétaire et plan d'investissement personnalisés, propulsés par l'IA (Groq).")
st.divider()

etape1, etape2, etape3 = st.tabs(
    ["Étape 1 — Situation & Documents", "Étape 2 — Profil de risque (optionnel)", "Étape 3 — Diagnostic & Plan"]
)


# ----------------------------------------------------------------------
# ÉTAPE 1 — IDENTITÉ, STATUT PROFESSIONNEL, CODE POSTAL, LOGEMENT, ETC.
# ----------------------------------------------------------------------
with etape1:
    st.subheader("Identité & Foyer")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        age = st.number_input("Âge *", min_value=18, max_value=100, value=30, step=1)
    with col_b:
        nb_adultes = st.number_input("Nombre d'adultes *", min_value=1, max_value=10, value=1, step=1)
    with col_c:
        nb_enfants = st.number_input("Nombre d'enfants *", min_value=0, max_value=10, value=0, step=1)

    col_d, col_e = st.columns(2)
    with col_d:
        situation_matrimoniale = st.selectbox(
            "Situation matrimoniale *",
            ["Célibataire", "Marié(e)", "PACSÉ(e)", "Concubinage / Union libre", "Divorcé(e)", "Veuf / Veuve"],
        )
    with col_e:
        nb_parts_fiscales = st.number_input(
            "Nombre de parts fiscales *",
            min_value=1.0,
            max_value=10.0,
            value=float(nb_adultes + (nb_enfants * 0.5)),
            step=0.5,
        )

    handicap = st.checkbox("Handicap ou frais médicaux récurrents importants dans le foyer")

    st.markdown("---")
    st.subheader("Statut Professionnel & Localisation")
    col_statut, col_cp = st.columns(2)

    with col_statut:
        statut_pro = st.selectbox(
            "Statut professionnel *",
            ["Salarié", "Fonctionnaire", "Retraité", "Demandeur d'emploi", "Entrepreneur / Indépendant"],
        )

        sous_statut = None
        duree_restante = None

        if statut_pro == "Salarié":
            sous_statut = st.selectbox("Type de contrat *", ["CDI", "CDD", "Intérim", "Stage", "Alternance / Apprentissage"])
            if sous_statut in ["CDD", "Intérim", "Stage"]:
                duree_restante = st.number_input("Mois restants sur le contrat *", min_value=1, max_value=36, value=6, step=1)
        elif statut_pro == "Entrepreneur / Indépendant":
            sous_statut = st.selectbox(
                "Forme juridique / Régime *",
                [
                    "Auto-entrepreneur / Micro-entreprise",
                    "EI / EIRL",
                    "Société (SASU, EURL, SARL, SAS)",
                    "Profession Libérale / TNS",
                ],
            )

    with col_cp:
        code_postal = st.text_input("Code Postal (France) *", value="75001", max_chars=5)
        # Détection automatique du coût de la vie selon les 2 premiers chiffres du code postal (département)
        dept = code_postal.strip()[:2]
        if dept in DEPARTEMENTS_COUT_ELEVE:
            zone_cout_vie = "Élevé"
            st.caption("📍 Zone à coût de la vie élevé détectée (ajustement automatique du budget).")
        else:
            zone_cout_vie = "Standard"
            st.caption("📍 Zone à coût de la vie standard.")

    st.markdown("---")
    st.subheader("Logement & Charges Essentielles")
    col_log1, col_log2, col_log3 = st.columns(3)

    with col_log1:
        statut_logement = st.selectbox(
            "Statut résidentiel *",
            ["Locataire", "Propriétaire avec crédit immo", "Propriétaire sans crédit", "Hébergé à titre gratuit"],
        )

    with col_log2:
        if statut_logement in ["Locataire", "Propriétaire avec crédit immo"]:
            label_logement = "Loyer mensuel (€) *" if statut_logement == "Locataire" else "Mensualité crédit immo (€) *"
            montant_logement = st.number_input(label_logement, min_value=0.0, value=700.0, step=50.0)
        else:
            montant_logement = 0.0
            st.info("Logement : 0 €/mois")

    with col_log3:
        autres_charges = st.number_input(
            "Autres charges essentielles (€/mois) *",
            help="Énergie, eau, assurances, alimentation, transports",
            min_value=0.0,
            value=400.0,
            step=50.0,
        )

    st.markdown("---")
    st.subheader("Dettes Hors Immobilier")
    col_det1, col_det2 = st.columns(2)
    with col_det1:
        montant_dettes = st.number_input("Capital total restant dû (€)", min_value=0.0, value=0.0, step=500.0)
    with col_det2:
        mensualite_dettes = st.number_input("Mensualité de remboursement (€/mois)", min_value=0.0, value=0.0, step=50.0)

    st.markdown("---")
    st.subheader("Bilan Patrimonial & Revenus")

    st.markdown("##### Fiche de paie (Optionnel)")
    fichier_paie = st.file_uploader(
        "Importer votre fiche de paie (PDF, 10 Mo max)",
        type=["pdf"],
        help="L'IA lira votre fiche de paie pour pré-remplir le salaire net.",
    )

    salaire_net_extrait = None
    if fichier_paie is not None:
        if fichier_paie.size > TAILLE_MAX_FICHIER_OCTETS:
            st.error(f"Le fichier dépasse la limite de {TAILLE_MAX_FICHIER_MO} Mo.")
        else:
            with st.spinner("Analyse du document PDF..."):
                texte = extraire_texte_pdf(fichier_paie)
                if texte:
                    salaire_net_extrait = extraire_salaire_net(texte)
                    if salaire_net_extrait:
                        st.success(f"Salaire net identifié : **{salaire_net_extrait:.2f} €**")

    col_rev1, col_rev2 = st.columns(2)
    with col_rev1:
        valeur_init_salaire = salaire_net_extrait if salaire_net_extrait else 2200.0
        salaire_net = st.number_input("Salaire net mensuel (€) *", min_value=0.0, value=valeur_init_salaire, step=100.0)
    with col_rev2:
        autres_revenus = st.number_input("Autres revenus mensuels (€)", help="Foncier, dividendes, primes", min_value=0.0, value=0.0, step=100.0)

    col_pat1, col_pat2 = st.columns(2)
    with col_pat1:
        epargne_dispo = st.number_input("Épargne de précaution disponible (€)", help="Comptes courants, Livret A, LDDS, LEP", min_value=0.0, value=5000.0, step=500.0)
    with col_pat2:
        epargne_lt = st.number_input("Épargne long terme existante (€)", help="PEA, Assurance-Vie, PER, CTO, Crypto, Immobilier", min_value=0.0, value=0.0, step=1000.0)

    produits_detenus = st.multiselect(
        "Produits financiers déjà détenus",
        ["Livret A / LDDS", "LEP", "PEA", "Assurance-Vie", "PER", "Compte-Titres (CTO)", "Crypto-actifs", "Immobilier locatif / SCPI"],
        default=["Livret A / LDDS"],
    )

    col_tmi, col_aide_tmi = st.columns([1, 2], vertical_alignment="center")
    with col_tmi:
        tmi_pct = st.selectbox("TMI (Tranche Marginale d'Imposition) *", [0, 11, 30, 41, 45], index=1)
    with col_aide_tmi:
        st.info(TEXTE_AIDE_TMI)


# ----------------------------------------------------------------------
# ÉTAPE 2 — PROFIL DE RISQUE (OPTIONNEL)
# ----------------------------------------------------------------------
with etape2:
    st.subheader("Évaluation du profil d'aversion au risque")
    activer_qcm = st.checkbox("Je souhaite remplir le questionnaire d'appétence au risque", value=True)

    reponses_qcm = {}
    niveau_connaissance = "Intermédiaire"

    if activer_qcm:
        for key, q in QUESTIONS_RISQUE.items():
            reponses_qcm[key] = st.radio(q["label"], list(q["options"].keys()), index=1)

        st.markdown("---")
        choix_connaissance = st.radio(
            "Comment évaluez-vous votre niveau en finance / investissement ?",
            list(CONNAISSANCES_FINANCIERES.keys()),
            index=1,
        )
        niveau_connaissance = CONNAISSANCES_FINANCIERES[choix_connaissance]
    else:
        reponses_qcm = {"horizon": None, "reaction_baisse": None, "preference_repartition": None}

    profil_risque, score_risque = calculer_profil_risque(reponses_qcm)


# ----------------------------------------------------------------------
# ÉTAPE 3 — DIAGNOSTIC & GENERATION GROQ
# ----------------------------------------------------------------------
with etape3:
    st.subheader("Diagnostic Budgétaire & Plan d'Investissement")

    if st.button("Lancer l'analyse IA", type="primary"):
        with st.spinner("Analyse patrimoniale et génération du rapport par Groq..."):
            # 1. Calcul Temps 1
            temps1 = calculer_budget_temps1(
                salaire_net=salaire_net,
                autres_revenus=autres_revenus,
                montant_logement=montant_logement,
                autres_charges_essentielles=autres_charges,
                mensualite_dette_hors_immo=mensualite_dettes,
                tmi_pct=tmi_pct,
                handicap=handicap,
                nb_personnes=nb_adultes + nb_enfants,
                zone_cout_vie=zone_cout_vie,
            )

            # 2. Calcul Temps 2
            temps2 = construire_allocation_temps2(
                capacite_epargne=temps1["capacite_epargne"],
                depenses_essentielles=temps1["depenses_essentielles"],
                epargne_disponible=epargne_dispo,
                age=age,
                tmi_pct=tmi_pct,
                profil_risque=profil_risque,
                statut_professionnel=statut_pro,
                sous_statut=sous_statut,
                duree_restante_mois=duree_restante,
            )

            # 3. Structuration des données
            identite_dict = {
                "age": age,
                "nb_adultes": nb_adultes,
                "nb_enfants": nb_enfants,
                "situation_matrimoniale": situation_matrimoniale,
                "nb_parts_fiscales": nb_parts_fiscales,
                "handicap": handicap,
            }
            emploi_dict = {
                "statut_professionnel": statut_pro,
                "sous_statut": sous_statut,
                "duree_restante_mois": duree_restante,
            }
            localisation_dict = {
                "code_postal": code_postal,
                "zone_cout_vie": zone_cout_vie,
            }
            logement_dict = {
                "statut": statut_logement,
                "montant": montant_logement,
                "autres_charges": autres_charges,
            }
            dettes_dict = {
                "montant_total": montant_dettes,
                "mensualite": mensualite_dettes,
            }
            patrimoine_dict = {
                "salaire_net": salaire_net,
                "autres_revenus": autres_revenus,
                "epargne_disponible": epargne_dispo,
                "epargne_long_terme": epargne_lt,
                "produits_detenus": ", ".join(produits_detenus) if produits_detenus else "Aucun",
                "tmi_pct": tmi_pct,
            }

            # 4. Appel Groq
            rapport_md = generer_rapport(
                identite=identite_dict,
                emploi=emploi_dict,
                localisation=localisation_dict,
                logement=logement_dict,
                dettes=dettes_dict,
                patrimoine=patrimoine_dict,
                temps1=temps1,
                temps2=temps2,
                profil_risque=profil_risque,
                score_risque=score_risque,
                niveau_connaissance=niveau_connaissance,
            )

            st.session_state.resultat = {
                "temps1": temps1,
                "temps2": temps2,
                "rapport": rapport_md,
            }

    # Affichage des résultats
