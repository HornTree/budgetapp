"""
Aurelion Wealth Management — Diagnostic Budgétaire & Plan d'Investissement
---------------------------------------------------------------------------
Cette application :

1. Recueille l'identité, le foyer, le logement et le bilan patrimonial
   complet de l'utilisateur (Étape 1).
2. Évalue, de façon optionnelle, un profil d'aversion au risque
   (Prudent / Équilibré / Dynamique) via un mini-QCM (Étape 2).
3. Extrait le texte d'une fiche de paie PDF (pdfplumber, 10 Mo max) et en
   déduit le salaire net via l'IA (Groq / llama-3.3-70b-versatile).
4. Calcule, en deux temps :
     - Temps 1 : la répartition du budget mensuel (essentiels, impôts,
       plaisirs, capacité d'épargne).
     - Temps 2 : l'allocation de la capacité d'épargne (matelas de
       sécurité, PER, investissement long terme), selon l'âge, la TMI,
       le patrimoine existant et le profil de risque.
5. Demande à Groq de rédiger un rapport Markdown structuré suivant
   strictement ce plan en deux temps (Étape 3).

Prérequis :
    pip install streamlit pdfplumber groq plotly pandas

Clé API :
    Ajoutez votre clé dans .streamlit/secrets.toml :
        GROQ_API_KEY = "votre_cle_api"

Limite d'upload :
    La taille maximale des fichiers importés (10 Mo) est fixée au niveau
    serveur via .streamlit/config.toml ([server] maxUploadSize = 10) et
    revérifiée manuellement dans le code par sécurité.
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
TAILLE_MAX_FICHIER_MO = 10
TAILLE_MAX_FICHIER_OCTETS = TAILLE_MAX_FICHIER_MO * 1024 * 1024

st.set_page_config(page_title="Aurelion Wealth Management", page_icon="€", layout="centered")


# ----------------------------------------------------------------------
# CLIENT GROQ
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def obtenir_client_groq():
    """
    Initialise (une seule fois, grâce au cache) le client Groq à partir de
    la clé API stockée dans st.secrets. Retourne None si la clé est
    absente, pour permettre à l'appelant d'afficher un message clair.
    """
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def appeler_groq(prompt: str) -> str:
    """
    Envoie un prompt au modèle Groq (llama-3.3-70b-versatile) et retourne
    le texte généré. Centralise la gestion d'erreurs pour les deux usages
    de l'application (extraction du salaire, rédaction du rapport).
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
    Demande à Groq d'extraire le "Salaire Net à payer" du texte de la
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
# MOTEUR DE CALCUL — TEMPS 1 : Répartition du budget mensuel
# ----------------------------------------------------------------------
def calculer_budget_temps1(
    salaire_net: float,
    autres_revenus: float,
    montant_logement: float,
    autres_charges_essentielles: float,
    mensualite_dette_hors_immo: float,
    tmi_pct: float,
    handicap: bool,
    nb_personnes: int,
) -> dict:
    """
    TEMPS 1 : ventile les revenus mensuels du foyer en 4 postes :

    1. Dépenses essentielles = logement (loyer ou mensualité crédit
       immobilier) + charges courantes (factures, alimentation,
       transports) + mensualité des crédits hors immobilier.
    2. Impôts & taxes = estimation simplifiée de l'impôt mensuel, obtenue
       en appliquant la Tranche Marginale d'Imposition (TMI) déclarée au
       revenu total. NB : il s'agit d'un ordre de grandeur (le calcul
       réel de l'impôt suit un barème progressif et un quotient familial)
       et non d'un calcul fiscal exact.
    3. Plaisirs / reste à vivre : part du solde consacrée aux dépenses
       variables et loisirs.
    4. Capacité d'épargne mensuelle : part du solde orientée vers
       l'épargne/investissement (Temps 2).

    Le partage entre "Plaisirs" et "Capacité d'épargne" est pondéré par
    le contexte du foyer : le taux d'épargne recommandé est réduit en cas
    de handicap/frais médicaux importants ou de foyer nombreux (plus de 3
    personnes), pour laisser davantage de marge de vie quotidienne.
    """
    revenu_total = round(salaire_net + autres_revenus, 2)
    depenses_essentielles = round(montant_logement + autres_charges_essentielles + mensualite_dette_hors_immo, 2)
    impots_mensuels = round(revenu_total * tmi_pct / 100, 2)

    reste_a_vivre_brut = round(revenu_total - depenses_essentielles - impots_mensuels, 2)
    deficit = reste_a_vivre_brut < 0
    reste_a_vivre = max(reste_a_vivre_brut, 0.0)

    taux_epargne = 0.50
    if handicap:
        taux_epargne = 0.30
    elif nb_personnes > 3:
        taux_epargne = 0.35

    capacite_epargne = round(reste_a_vivre * taux_epargne, 2)
    plaisirs = round(reste_a_vivre - capacite_epargne, 2)

    return {
        "revenu_total": revenu_total,
        "depenses_essentielles": depenses_essentielles,
        "impots_mensuels": impots_mensuels,
        "reste_a_vivre_brut": reste_a_vivre_brut,
        "deficit": deficit,
        "plaisirs": plaisirs,
        "capacite_epargne": capacite_epargne,
    }


# ----------------------------------------------------------------------
# MOTEUR DE CALCUL — TEMPS 2 : Allocation de la capacité d'épargne
# ----------------------------------------------------------------------
def construire_allocation_temps2(
    capacite_epargne: float,
    depenses_essentielles: float,
    epargne_disponible: float,
    age: int,
    tmi_pct: float,
    profil_risque: str,
) -> dict:
    """
    TEMPS 2 : répartit la capacité d'épargne mensuelle (issue du Temps 1)
    entre trois blocs, dans l'ordre de priorité patrimonial classique :

    1. Matelas de sécurité : alimentation des livrets réglementés
       jusqu'à atteindre une cible de 3 à 6 mois de charges essentielles
       (cible resserrée à 3 mois pour un profil Dynamique, élargie à 6
       mois pour un profil Prudent). Priorité forte tant que l'épargne
       disponible couvre moins de 3 mois de charges.
    2. PER (optimisation fiscale/retraite) : pertinent surtout si la TMI
       est élevée (≥ 30%) et si l'âge laisse un horizon de placement
       raisonnable avant la retraite (< 60 ans), le PER bloquant les
       fonds jusqu'à cet âge.
    3. Investissement long terme (PEA / Assurance-Vie / ETF / SCPI) :
       reçoit le solde restant après matelas et PER.
    """
    if depenses_essentielles > 0:
        mois_couverture_actuelle = round(epargne_disponible / depenses_essentielles, 1)
    else:
        mois_couverture_actuelle = 99.0  # pas de charges essentielles déclarées

    cible_matelas_mois = {"Prudent": 6, "Équilibré": 4.5, "Dynamique": 3, "Non renseigné": 4.5}.get(profil_risque, 4.5)
    montant_cible_matelas = depenses_essentielles * cible_matelas_mois
    manque_matelas = max(montant_cible_matelas - epargne_disponible, 0.0)

    if capacite_epargne <= 0:
        return {
            "mois_couverture_actuelle": mois_couverture_actuelle,
            "cible_matelas_mois": cible_matelas_mois,
            "montant_matelas": 0.0,
            "montant_per": 0.0,
            "montant_invest_long_terme": 0.0,
        }

    # Taux d'effort vers le matelas selon le niveau de couverture actuel
    if mois_couverture_actuelle < 3:
        taux_matelas = 0.60
    elif mois_couverture_actuelle < cible_matelas_mois:
        taux_matelas = 0.30
    else:
        taux_matelas = 0.05  # entretien minimal du matelas déjà constitué

    montant_matelas = round(capacite_epargne * taux_matelas, 2)
    if manque_matelas <= 0:
        # Le matelas cible est déjà atteint : on limite l'effort à un entretien symbolique
        montant_matelas = round(min(montant_matelas, capacite_epargne * 0.05), 2)

    reste_apres_matelas = round(capacite_epargne - montant_matelas, 2)

    # Pertinence du PER : TMI élevée et horizon retraite pas trop proche/déjà atteint
    if tmi_pct >= 30 and age < 60:
        taux_per = 0.30 if tmi_pct >= 41 else 0.20
    else:
        taux_per = 0.0
    montant_per = round(reste_apres_matelas * taux_per, 2)

    montant_invest_long_terme = round(reste_apres_matelas - montant_per, 2)

    return {
        "mois_couverture_actuelle": mois_couverture_actuelle,
        "cible_matelas_mois": cible_matelas_mois,
        "montant_matelas": montant_matelas,
        "montant_per": montant_per,
        "montant_invest_long_terme": montant_invest_long_terme,
    }


# ----------------------------------------------------------------------
# GÉNÉRATION DU RAPPORT MARKDOWN VIA GROQ
# ----------------------------------------------------------------------
def generer_rapport(
    identite: dict,
    logement: dict,
    dettes: dict,
    patrimoine: dict,
    temps1: dict,
    temps2: dict,
    profil_risque: str,
    score_risque: float,
) -> str:
    """
    Construit un prompt détaillé décrivant l'ensemble de la situation du
    foyer (identité, logement, dettes, patrimoine) ainsi que les montants
    calculés par le moteur Temps 1 / Temps 2, puis demande à Groq de
    rédiger un rapport Markdown structuré suivant strictement ce plan en
    deux temps.
    """
    prompt = f"""Tu es un conseiller en gestion de patrimoine, bienveillant, pédagogue et rigoureux.
Rédige un rapport en Markdown clair et structuré à partir des données suivantes.
Précise en une phrase, avant le Temps 1, que ce diagnostic est indicatif et ne remplace pas l'avis d'un conseiller en gestion de patrimoine agréé.

IDENTITÉ & FOYER :
- Âge : {identite['age']} ans
- Situation matrimoniale : {identite['situation_matrimoniale']}
- Nombre de parts fiscales : {identite['nb_parts_fiscales']}
- Nombre d'adultes : {identite['nb_adultes']} / Nombre d'enfants : {identite['nb_enfants']}
- Handicap / frais médicaux importants : {"Oui" if identite['handicap'] else "Non"}

LOGEMENT :
- Statut : {logement['statut']}
- Montant mensuel logement (loyer ou mensualité crédit) : {logement['montant']:.2f} €
- Autres charges essentielles mensuelles (factures, alimentation, transport) : {logement['autres_charges']:.2f} €

DETTES HORS IMMOBILIER :
- Montant total restant dû : {dettes['montant_total']:.2f} €
- Mensualité de remboursement : {dettes['mensualite']:.2f} €

BILAN PATRIMONIAL :
- Salaire net mensuel : {patrimoine['salaire_net']:.2f} €
- Autres revenus mensuels (fonciers, dividendes, primes) : {patrimoine['autres_revenus']:.2f} €
- Épargne disponible (comptes courants, Livret A, LDDS, LEP) : {patrimoine['epargne_disponible']:.2f} €
- Épargne long terme existante (PEA, Assurance-Vie, PER, CTO, Crypto, Immobilier) : {patrimoine['epargne_long_terme']:.2f} €
- Produits déjà détenus : {patrimoine['produits_detenus']}
- Tranche Marginale d'Imposition (TMI) déclarée : {patrimoine['tmi_pct']:.0f}%
- Profil d'aversion au risque : {profil_risque} (score {score_risque}/3 ; "Non renseigné" si le QCM optionnel n'a pas été rempli)

RÉSULTATS DU MOTEUR DE CALCUL — TEMPS 1 (Répartition du budget mensuel) :
- Revenu total mensuel : {temps1['revenu_total']:.2f} €
- Dépenses essentielles (logement + charges + crédits hors immo) : {temps1['depenses_essentielles']:.2f} €
- Impôts & taxes estimés (approximation via la TMI) : {temps1['impots_mensuels']:.2f} €
- Plaisirs / reste à vivre : {temps1['plaisirs']:.2f} €
- Capacité d'épargne mensuelle calculée : {temps1['capacite_epargne']:.2f} €
- Situation de déficit budgétaire ce mois-ci : {"Oui, le solde brut est de " + f"{temps1['reste_a_vivre_brut']:.2f} €" if temps1['deficit'] else "Non"}

RÉSULTATS DU MOTEUR DE CALCUL — TEMPS 2 (Allocation de la capacité d'épargne) :
- Couverture actuelle du matelas de sécurité : {temps2['mois_couverture_actuelle']:.1f} mois de charges essentielles (cible recommandée : {temps2['cible_matelas_mois']} mois)
- Allocation Matelas de sécurité : {temps2['montant_matelas']:.2f} €
- Allocation PER (optimisation fiscale/retraite) : {temps2['montant_per']:.2f} €
- Allocation Investissement long terme (PEA / Assurance-Vie / ETF / SCPI) : {temps2['montant_invest_long_terme']:.2f} €

CONSIGNES DE RÉDACTION :
Structure le rapport en suivant EXACTEMENT ce plan, avec des titres Markdown (##) :

## Temps 1 — Répartition du budget mensuel et diagnostic de santé financière
Un paragraphe de diagnostic (taille du foyer, présence de dettes, contexte médical si pertinent, situation de déficit éventuelle), puis un tableau Markdown (colonnes : Poste | Montant en € | % du revenu total) reprenant les 4 postes du Temps 1.

## Temps 2 — Plan d'investissement personnalisé
Un tableau Markdown (colonnes : Allocation | Montant en € | % de la capacité d'épargne) reprenant les 3 blocs du Temps 2, avec une phrase expliquant pourquoi cette allocation est cohérente avec l'âge, la TMI, le patrimoine existant et le profil de risque.
Ensuite, une liste numérotée de 3 conseils pratiques et bienveillants, puis une "Feuille de route" numérotée de 3 à 5 actions concrètes à réaliser dès ce mois-ci.

Réponds uniquement avec le rapport en Markdown, sans phrase d'introduction avant le titre.
"""
    return appeler_groq(prompt)


# ----------------------------------------------------------------------
# INITIALISATION DE L'ÉTAT DE SESSION
# ----------------------------------------------------------------------
if "resultat" not in st.session_state:
    st.session_state.resultat = None  # stockera (temps1, temps2, rapport) après génération


# ----------------------------------------------------------------------
# EN-TÊTE
# ----------------------------------------------------------------------
st.title("Aurelion Wealth Management")
st.caption(
    "Renseignez votre situation complète, importez votre fiche de paie, et obtenez "
    "un diagnostic budgétaire et un plan d'investissement personnalisés (IA Groq)."
)

etape1, etape2, etape3 = st.tabs(
    ["Étape 1 — Situation & Documents", "Étape 2 — Profil de risque (optionnel)", "Étape 3 — Diagnostic & Plan"]
)


# ----------------------------------------------------------------------
# ÉTAPE 1 — IDENTITÉ, FOYER, LOGEMENT, DETTES, PATRIMOINE, DOCUMENTS
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
            ["Célibataire", "Marié(e)", "Pacsé(e)", "Divorcé(e)", "Veuf(ve)"],
        )
    with col_e:
        nb_parts_fiscales = st.number_input(
            "Nombre de parts fiscales *", min_value=1.0, max_value=10.0, value=1.0, step=0.5, format="%.1f"
        )

    handicap = st.checkbox("Handicap / Frais médicaux importants")

    st.divider()
    st.subheader("Logement & Charges principales")
    statut_logement = st.selectbox(
        "Statut de résidence *",
        [
            "Locataire",
            "Propriétaire avec crédit immobilier en cours",
            "Propriétaire sans crédit / Occupant à titre gratuit",
        ],
    )
    if statut_logement == "Locataire":
        label_montant_logement = "Loyer mensuel (€) *"
        montant_logement = st.number_input(label_montant_logement, min_value=0.0, value=0.0, step=10.0, format="%.2f")
    elif statut_logement == "Propriétaire avec crédit immobilier en cours":
        label_montant_logement = "Mensualité du crédit immobilier (€) *"
        montant_logement = st.number_input(label_montant_logement, min_value=0.0, value=0.0, step=10.0, format="%.2f")
    else:
        st.info("Aucun loyer ni mensualité de crédit à renseigner pour ce statut.")
        montant_logement = 0.0

    autres_charges_essentielles = st.number_input(
        "Autres charges essentielles mensuelles (factures, alimentation, transports) *",
        min_value=0.0, value=0.0, step=10.0, format="%.2f",
    )

    st.divider()
    st.subheader("Dettes hors immobilier")
    col_f, col_g = st.columns(2)
    with col_f:
        montant_dette = st.number_input(
            "Montant total des dettes hors immo (conso, auto, personnelles) (€)",
            min_value=0.0, value=0.0, step=100.0, format="%.2f",
        )
    with col_g:
        mensualite_dette = st.number_input(
            "Mensualité de remboursement (€)", min_value=0.0, value=0.0, step=10.0, format="%.2f"
        )

    st.divider()
    st.subheader("Bilan patrimonial & financier global")
    col_h, col_i = st.columns(2)
    with col_h:
        autres_revenus = st.number_input(
            "Autres revenus mensuels (fonciers, dividendes, primes) (€)",
            min_value=0.0, value=0.0, step=10.0, format="%.2f",
        )
    with col_i:
        tmi_pct = st.selectbox(
            "Tranche Marginale d'Imposition (TMI) estimée *",
            options=[0, 11, 30, 41, 45],
            index=1,
            format_func=lambda v: f"{v}%",
        )

    col_j, col_k = st.columns(2)
    with col_j:
        epargne_disponible = st.number_input(
            "Épargne disponible (comptes courants, Livret A, LDDS, LEP) (€)",
            min_value=0.0, value=0.0, step=100.0, format="%.2f",
        )
    with col_k:
        epargne_long_terme = st.number_input(
            "Épargne long terme existante (PEA, Assurance-Vie, PER, CTO, Crypto, Immobilier) (€)",
            min_value=0.0, value=0.0, step=100.0, format="%.2f",
        )

    produits_detenus = st.multiselect(
        "Produits déjà détenus (facultatif)",
        ["Assurance Vie", "PEA", "Compte-titres ordinaire (CTO)", "PER", "Immobilier locatif", "Crypto-actifs", "Aucun"],
    )

    st.divider()
    st.subheader("Fiche de paie")
    st.caption(f"Format PDF uniquement, {TAILLE_MAX_FICHIER_MO} Mo maximum par fichier.")
    fichier_pdf = st.file_uploader("Importer votre fiche de paie (PDF)", type=["pdf"])

    if fichier_pdf is not None:
        if fichier_pdf.size > TAILLE_MAX_FICHIER_OCTETS:
            st.error(
                f"Le fichier dépasse la limite autorisée de {TAILLE_MAX_FICHIER_MO} Mo "
                f"({fichier_pdf.size / (1024 * 1024):.1f} Mo). Merci d'importer un fichier plus léger."
            )
            fichier_pdf = None
        else:
            st.success("Fichier importé. Rendez-vous dans l'Étape 3 pour lancer l'analyse.")

    st.info("Passez ensuite à l'Étape 2 (facultative) ou directement à l'Étape 3.")


# ----------------------------------------------------------------------
# ÉTAPE 2 — PROFIL DE RISQUE (OPTIONNEL)
# ----------------------------------------------------------------------
with etape2:
    st.subheader("Votre rapport au risque (facultatif)")
    st.caption(
        "Ces questions permettent d'affiner l'allocation entre matelas de sécurité, "
        "PER et investissement long terme. Vous pouvez laisser ces champs vides : "
        "un profil équilibré par défaut sera utilisé."
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
            st.success(f"Profil de risque estimé : {profil_risque} (score {score_risque}/3)")
        else:
            st.warning("Répondez à au moins une question pour estimer votre profil.")
    else:
        profil_risque, score_risque = "Non renseigné", 0.0
        st.info("QCM non renseigné : un profil Équilibré par défaut sera utilisé pour les recommandations.")


# ----------------------------------------------------------------------
# ÉTAPE 3 — DIAGNOSTIC & PLAN D'INVESTISSEMENT
# ----------------------------------------------------------------------
with etape3:
    st.subheader("Générer mon diagnostic budgétaire et mon plan d'investissement")

    lancer = st.button("Lancer l'analyse IA", use_container_width=True, type="primary")

    if lancer:
        if fichier_pdf is None:
            st.warning("Merci d'importer une fiche de paie valide (Étape 1) avant de continuer.")
        else:
            with st.spinner("Lecture du PDF en cours..."):
                texte_pdf = extraire_texte_pdf(fichier_pdf)

            if not texte_pdf:
                st.error("Le texte n'a pas pu être extrait du PDF (fichier scanné/image ou vide).")
            else:
                with st.expander("Voir le texte extrait de la fiche de paie"):
                    st.text(texte_pdf)

                with st.spinner("Extraction du salaire net via Groq..."):
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
                    st.success(f"Salaire net détecté : {salaire_net:.2f} €")

                    nb_personnes = nb_adultes + nb_enfants

                    temps1 = calculer_budget_temps1(
                        salaire_net=salaire_net,
                        autres_revenus=autres_revenus,
                        montant_logement=montant_logement,
                        autres_charges_essentielles=autres_charges_essentielles,
                        mensualite_dette_hors_immo=mensualite_dette,
                        tmi_pct=tmi_pct,
                        handicap=handicap,
                        nb_personnes=nb_personnes,
                    )

                    temps2 = construire_allocation_temps2(
                        capacite_epargne=temps1["capacite_epargne"],
                        depenses_essentielles=temps1["depenses_essentielles"],
                        epargne_disponible=epargne_disponible,
                        age=age,
                        tmi_pct=tmi_pct,
                        profil_risque=profil_risque,
                    )

                    identite = {
                        "age": age,
                        "situation_matrimoniale": situation_matrimoniale,
                        "nb_parts_fiscales": nb_parts_fiscales,
                        "nb_adultes": nb_adultes,
                        "nb_enfants": nb_enfants,
                        "handicap": handicap,
                    }
                    logement = {
                        "statut": statut_logement,
                        "montant": montant_logement,
                        "autres_charges": autres_charges_essentielles,
                    }
                    dettes = {"montant_total": montant_dette, "mensualite": mensualite_dette}
                    patrimoine = {
                        "salaire_net": salaire_net,
                        "autres_revenus": autres_revenus,
                        "epargne_disponible": epargne_disponible,
                        "epargne_long_terme": epargne_long_terme,
                        "produits_detenus": ", ".join(produits_detenus) if produits_detenus else "Aucun déclaré",
                        "tmi_pct": tmi_pct,
                    }

                    with st.spinner("Rédaction du rapport personnalisé via Groq..."):
                        rapport = generer_rapport(
                            identite=identite,
                            logement=logement,
                            dettes=dettes,
                            patrimoine=patrimoine,
                            temps1=temps1,
                            temps2=temps2,
                            profil_risque=profil_risque,
                            score_risque=score_risque,
                        )

                    st.session_state.resultat = (temps1, temps2, rapport)
                else:
                    st.info("Renseignez un salaire net valide pour lancer le calcul du budget.")

    # --- Affichage du résultat le plus récent, s'il existe ---
    if st.session_state.resultat is not None:
        temps1, temps2, rapport = st.session_state.resultat

        if temps1["deficit"]:
            st.error(
                f"Attention : votre budget est en déficit ce mois-ci "
                f"(solde brut de {temps1['reste_a_vivre_brut']:.2f} € avant tout arbitrage)."
            )

        st.divider()
        st.subheader("Temps 1 — Répartition du budget mensuel")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dépenses essentielles", f"{temps1['depenses_essentielles']:.2f} €")
        col2.metric("Impôts & taxes (estimation)", f"{temps1['impots_mensuels']:.2f} €")
        col3.metric("Plaisirs / reste à vivre", f"{temps1['plaisirs']:.2f} €")
        col4.metric("Capacité d'épargne", f"{temps1['capacite_epargne']:.2f} €")

        df_temps1 = pd.DataFrame(
            {
                "Poste": ["Dépenses essentielles", "Impôts & taxes", "Plaisirs / reste à vivre", "Capacité d'épargne"],
                "Montant (€)": [
                    temps1["depenses_essentielles"],
                    temps1["impots_mensuels"],
                    temps1["plaisirs"],
                    temps1["capacite_epargne"],
                ],
            }
        )
        fig1 = go.Figure(
            data=[
                go.Bar(
                    x=df_temps1["Poste"],
                    y=df_temps1["Montant (€)"],
                    text=df_temps1["Montant (€)"].map(lambda v: f"{v:.0f} €"),
                    textposition="outside",
                    marker_color=["#4C78A8", "#B279A2", "#54A24B", "#F58518"],
                )
            ]
        )
        fig1.update_layout(
            title="Répartition du revenu total mensuel",
            yaxis_title="Montant (€)",
            showlegend=False,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()
        st.subheader("Temps 2 — Plan d'investissement personnalisé")
        st.caption(
            f"Couverture actuelle du matelas de sécurité : {temps2['mois_couverture_actuelle']:.1f} mois de "
            f"charges essentielles (cible recommandée : {temps2['cible_matelas_mois']} mois)."
        )

        col5, col6, col7 = st.columns(3)
        col5.metric("Matelas de sécurité", f"{temps2['montant_matelas']:.2f} €")
        col6.metric("PER (fiscal / retraite)", f"{temps2['montant_per']:.2f} €")
        col7.metric("Investissement long terme", f"{temps2['montant_invest_long_terme']:.2f} €")

        df_temps2 = pd.DataFrame(
            {
                "Allocation": ["Matelas de sécurité", "PER", "Investissement long terme"],
                "Montant (€)": [
                    temps2["montant_matelas"],
                    temps2["montant_per"],
                    temps2["montant_invest_long_terme"],
                ],
            }
        )
        fig2 = go.Figure(
            data=[
                go.Bar(
                    x=df_temps2["Allocation"],
                    y=df_temps2["Montant (€)"],
                    text=df_temps2["Montant (€)"].map(lambda v: f"{v:.0f} €"),
                    textposition="outside",
                    marker_color=["#4C78A8", "#F58518", "#54A24B"],
                )
            ]
        )
        fig2.update_layout(
            title="Allocation de la capacité d'épargne mensuelle",
            yaxis_title="Montant (€)",
            showlegend=False,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Rapport détaillé")
        if rapport:
            st.markdown(rapport)
        else:
            st.error("Le rapport n'a pas pu être généré. Vérifiez votre clé API Groq.")

        st.caption(
            "Ce diagnostic est fourni à titre indicatif et ne constitue pas un conseil en "
            "investissement personnalisé au sens réglementaire. Consultez un conseiller en "
            "gestion de patrimoine agréé avant toute décision."
        )
    elif not lancer:
        st.info("Complétez l'Étape 1 (et éventuellement l'Étape 2), puis cliquez sur \"Lancer l'analyse IA\".")
