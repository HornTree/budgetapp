"""
Aurelion Wealth Management — Diagnostic Budgétaire & Plan d'Investissement
---------------------------------------------------------------------------
Cette application :

1. Recueille l'identité, le foyer, le statut professionnel (salarié,
   fonctionnaire, retraité, demandeur d'emploi, entrepreneur/indépendant
   avec sous-statut), la région, le logement et le bilan patrimonial
   complet de l'utilisateur (Étape 1). Le salaire net peut être renseigné
   soit via l'import d'une fiche de paie PDF (10 Mo max), soit par saisie
   manuelle directe.
2. Évalue, de façon optionnelle, un profil d'aversion au risque
   (Prudent / Équilibré / Dynamique) via un mini-QCM resserré à 3
   questions non redondantes, complété d'une question de connaissances
   financières utilisée pour adapter le ton du rapport (Étape 2).
3. Si une fiche de paie a été importée, en extrait le texte (pdfplumber)
   puis en déduit le salaire net via l'IA (Groq / llama-3.3-70b-versatile).
4. Calcule, en deux temps :
     - Temps 1 : Reste à Vivre Réel = Revenu - (Charges fixes + Impôts).
       La catégorie "Plaisirs" est plafonnée à un pourcentage raisonnable
       du revenu (15 à 25%, ajusté selon la région et le foyer) : tout
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
CHEMIN_LOGO = "assets/aurelion_logo.jpg"

# Régions dont le coût de la vie est jugé structurellement élevé (loyers,
# transport, alimentation). Simplification volontaire : ce n'est pas un
# indice INSEE précis, seulement un facteur d'ajustement qualitatif pour
# le plafond "Plaisirs" et pour le contexte donné à l'IA.
REGIONS = [
    "Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", "Centre-Val de Loire",
    "Corse", "Grand Est", "Hauts-de-France", "Île-de-France", "Normandie",
    "Nouvelle-Aquitaine", "Occitanie", "Pays de la Loire", "Provence-Alpes-Côte d'Azur",
    "Guadeloupe", "Martinique", "Guyane", "La Réunion", "Mayotte",
]
REGIONS_COUT_ELEVE = {"Île-de-France", "Provence-Alpes-Côte d'Azur", "Corse"}

TEXTE_AIDE_TMI = (
    "La Tranche Marginale d'Imposition (TMI) est le taux appliqué à votre dernière tranche "
    "de revenus, et non le pourcentage global payé sur vos revenus (Taux Moyen). "
    "Elle est essentielle pour calculer l'avantage fiscal d'un PER."
)

st.set_page_config(page_title="Aurelion Wealth Management", page_icon=CHEMIN_LOGO, layout="centered")

# ----------------------------------------------------------------------
# IDENTITÉ VISUELLE — Charte bleu marine / or, style minimaliste et sobre
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    h1, h2, h3 { color: #0A1F44; font-weight: 600; letter-spacing: 0.3px; }
    hr { border: none; border-top: 1px solid #C9A44C; margin: 1.4em 0; }
    div.stButton > button[kind="primary"] {
        background-color: #C9A44C; color: #0A1F44; border: none; font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover { background-color: #B8943F; color: #FFFFFF; }
    div[data-testid="stMetricValue"] { color: #0A1F44; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #C9A44C; }
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
# PROFIL DE RISQUE — Mini-QCM optionnel (resserré, sans redondance)
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
    de risque, puis en déduit un profil textuel : Prudent, Équilibré ou
    Dynamique. Retourne ("Non renseigné", 0.0) si aucune réponse.
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
    TEMPS 1 : calcule le Reste à Vivre Réel = Revenu - (Charges fixes +
    Impôts), puis ventile les revenus mensuels du foyer en 4 postes :

    1. Dépenses essentielles = logement + charges courantes + crédits
       hors immobilier.
    2. Impôts & taxes = estimation simplifiée via la TMI déclarée
       (ordre de grandeur, pas un calcul fiscal exact au barème réel).
    3. Plaisirs / reste à vivre : PLAFONNÉ à un pourcentage raisonnable
       du revenu total (20% par défaut, 25% en zone à coût de la vie
       élevé, réduit en cas de handicap/frais médicaux ou de foyer
       nombreux). Corrige le biais où un foyer sans charges importantes
       se voyait attribuer un budget loisirs disproportionné.
    4. Capacité d'épargne mensuelle : reçoit TOUT l'excédent du reste à
       vivre au-delà du plafond "Plaisirs" — l'argent non consommé n'est
       plus perdu dans une case "loisirs" surdimensionnée, il finance
       systématiquement le Temps 2 (épargne/investissement).
    """
    revenu_total = round(salaire_net + autres_revenus, 2)
    depenses_essentielles = round(montant_logement + autres_charges_essentielles + mensualite_dette_hors_immo, 2)
    impots_mensuels = round(revenu_total * tmi_pct / 100, 2)

    reste_a_vivre_brut = round(revenu_total - depenses_essentielles - impots_mensuels, 2)
    deficit = reste_a_vivre_brut < 0
    reste_a_vivre = max(reste_a_vivre_brut, 0.0)

    # Plafond "Plaisirs" en % du revenu total (et non du reste à vivre) :
    # c'est ce qui empêche un excédent important de gonfler les loisirs.
    taux_plafond_plaisir = 0.25 if zone_cout_vie == "Élevé" else 0.20
    if handicap:
        taux_plafond_plaisir -= 0.05
    elif nb_personnes > 3:
        taux_plafond_plaisir -= 0.03
    taux_plafond_plaisir = max(taux_plafond_plaisir, 0.10)

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
def obtenir_bonus_stabilite(statut_professionnel: str, sous_statut: str | None, duree_restante_mois: float | None) -> float:
    """
    Détermine, en mois de charges essentielles supplémentaires, la
    majoration de la cible de matelas de sécurité selon la stabilité du
    statut professionnel (CDI stable -> 0, statut précaire -> majoré).
    """
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
    """
    TEMPS 2 : répartit la capacité d'épargne mensuelle (issue du Temps 1)
    en 3 blocs, dans l'ordre de priorité patrimonial classique :

    1. Épargne de précaution / Secours (livrets réglementés) : cible de
       3 à 6 mois de charges essentielles selon le profil de risque,
       majorée pour les statuts professionnels précaires (CDD proche de
       son terme, intérim, stage, indépendant, demandeur d'emploi) et
       réduite pour les statuts jugés très stables (fonctionnaire,
       retraité).
    2. PER (optimisation fiscale/retraite) : pertinent si la TMI est
       élevée (≥ 30%) et l'âge laisse un horizon avant la retraite
       (< 60 ans). Toujours nul pour un statut "Retraité" (déjà en phase
       de décaissement).
    3. Investissement long terme (PEA / Assurance-Vie / ETF / SCPI) :
       reçoit le solde. Une part indicative "dynamique" (actions/ETF)
       vs "défensive" (fonds euro/obligataire) est suggérée selon le
       profil de risque et l'âge, à titre de repère pour l'IA — la
       répartition précise entre supports est détaillée dans le rapport.
    """
    if depenses_essentielles > 0:
        mois_couverture_actuelle = round(epargne_disponible / depenses_essentielles, 1)
    else:
        mois_couverture_actuelle = 99.0

    base_cible = {"Prudent": 6, "Équilibré": 4.5, "Dynamique": 3, "Non renseigné": 4.5}.get(profil_risque, 4.5)
    bonus_stabilite = obtenir_bonus_stabilite(statut_professionnel, sous_statut, duree_restante_mois)
    cible_matelas_mois = min(max(base_cible + bonus_stabilite, 3), 9)

    montant_cible_matelas = depenses_essentielles * cible_matelas_mois
    manque_matelas = max(montant_cible_matelas - epargne_disponible, 0.0)

    # Part "dynamique" recommandée au sein du bloc investissement long
    # terme, à titre indicatif pour la rédaction du rapport (pas une
    # sous-répartition monétaire figée par le moteur de calcul).
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
            "montant_matelas": 0.0,
            "montant_per": 0.0,
            "montant_invest_long_terme": 0.0,
            "part_dynamique_pct": part_dynamique_pct,
        }

    if mois_couverture_actuelle < 3:
        taux_matelas = 0.60
    elif mois_couverture_actuelle < cible_matelas_mois:
        taux_matelas = 0.30
    else:
        taux_matelas = 0.05

    montant_matelas = round(capacite_epargne * taux_matelas, 2)
    if manque_matelas <= 0:
        montant_matelas = round(min(montant_matelas, capacite_epargne * 0.05), 2)

    reste_apres_matelas = round(capacite_epargne - montant_matelas, 2)

    if statut_professionnel == "Retraité":
        taux_per = 0.0
    elif tmi_pct >= 30 and age < 60:
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
    """
    Construit un prompt détaillé décrivant l'ensemble de la situation du
    foyer ainsi que les montants calculés par le moteur Temps 1 / Temps 2,
    puis demande à Groq de rédiger un rapport Markdown concis suivant
    strictement ce plan en deux temps, avec un plan d'investissement
    structuré nommant des supports concrets et adapté au statut
    professionnel.
    """
    if emploi["sous_statut"]:
        detail_emploi = f"{emploi['statut_professionnel']} — {emploi['sous_statut']}"
    else:
        detail_emploi = emploi["statut_professionnel"]
    if emploi["duree_restante_mois"] is not None:
        detail_emploi += f" ({emploi['duree_restante_mois']:.0f} mois restants)"

    localisation_txt = localisation["region"]
    if localisation["code_postal"]:
        localisation_txt += f" (code postal {localisation['code_postal']})"

    prompt = f"""Tu es un conseiller en gestion de patrimoine, direct et rigoureux. Ton style est concis : pas de paragraphes longs, va à l'essentiel.
Adapte ton niveau de vocabulaire au niveau de connaissances financières de l'utilisateur : {niveau_connaissance} (vulgarise davantage si Novice, sois plus technique si Expert).
Adapte tes conseils retraite/prévoyance et ta gestion de la trésorerie selon le statut professionnel : si Entrepreneur/Indépendant, aborde la trésorerie de précaution professionnelle, la prévoyance (arrêt maladie/invalidité) et le PER adapté à des revenus irréguliers ; si Retraité, ne recommande jamais de PER (déjà en phase de décaissement) mais privilégie la transmission et l'assurance-vie ; si Demandeur d'emploi, priorise la reconstitution du matelas de sécurité avant tout investissement ; si Fonctionnaire, tu peux mentionner la Préfon-Retraite en complément du PER.

Rédige un rapport en Markdown à partir des données suivantes.
Commence par une phrase (pas plus) précisant que ce diagnostic est indicatif et ne remplace pas l'avis d'un conseiller en gestion de patrimoine agréé.

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

RÉSULTATS DU MOTEUR DE CALCUL — TEMPS 1 (Reste à Vivre Réel & répartition) :
- Revenu total mensuel : {temps1['revenu_total']:.2f} €
- Dépenses essentielles : {temps1['depenses_essentielles']:.2f} €
- Impôts & taxes estimés (approximation via la TMI) : {temps1['impots_mensuels']:.2f} €
- Plaisirs (plafonnés à {temps1['taux_plafond_plaisir']*100:.0f}% du revenu total) : {temps1['plaisirs']:.2f} €
- Capacité d'épargne mensuelle calculée (reçoit tout l'excédent au-delà du plafond Plaisirs) : {temps1['capacite_epargne']:.2f} €
- Situation de déficit budgétaire ce mois-ci : {"Oui, solde brut de " + f"{temps1['reste_a_vivre_brut']:.2f} €" if temps1['deficit'] else "Non"}

RÉSULTATS DU MOTEUR DE CALCUL — TEMPS 2 (Allocation de la capacité d'épargne) :
- Couverture actuelle du matelas de sécurité : {temps2['mois_couverture_actuelle']:.1f} mois de charges essentielles (cible : {temps2['cible_matelas_mois']} mois, ajustée selon le profil de risque et le statut professionnel)
- Allocation Épargne de précaution : {temps2['montant_matelas']:.2f} €
- Allocation PER (optimisation fiscale/retraite) : {temps2['montant_per']:.2f} €
- Allocation Investissement long terme : {temps2['montant_invest_long_terme']:.2f} € (dont environ {temps2['part_dynamique_pct']:.0f}% à orienter vers des supports dynamiques, le reste vers des supports défensifs)

CONSIGNES DE RÉDACTION (reste concis à chaque étape) :

## Temps 1 — Reste à vivre réel et répartition du budget mensuel
Diagnostic en 2-3 phrases MAXIMUM (taille du foyer, dettes, contexte médical si pertinent, déficit éventuel).
Tableau Markdown (Poste | Montant en € | % du revenu total) reprenant les 4 postes.
2 à 3 puces courtes expliquant POURQUOI cette répartition, en particulier pourquoi les Plaisirs sont plafonnés et où va l'excédent (pas de paragraphe).

## Temps 2 — Plan d'investissement personnalisé
Tableau Markdown (Allocation | Montant en € | % de la capacité d'épargne) reprenant les 3 lignes du Temps 2.
Un court paragraphe "Contexte de marché" (3-4 phrases maximum) donnant des repères généraux actuels par classe d'actifs (taux des livrets réglementés, fonds euro, tendances actions/obligations) — précise explicitement que ce sont des repères généraux et non une analyse de marché en temps réel, à vérifier avant toute décision.
Section "Dans quoi investir concrètement" : pour chaque ligne du Temps 2 (hors épargne de précaution), nomme 1 à 2 supports précis (ex. ETF monde capitalisant, SCPI de rendement, fonds euro d'assurance-vie, PER en unités de compte, obligations d'État) avec une justification d'une ligne chacun, tenant compte de l'âge, du profil de risque, du statut professionnel, des produits déjà détenus et de la part dynamique/défensive indiquée.
"Feuille de route" : liste numérotée de 3 à 5 actions concrètes à réaliser dès ce mois-ci.

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
col_logo, col_titre = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    st.image(CHEMIN_LOGO, width=88)
with col_titre:
    st.markdown("<h1 style='margin-bottom:0;'>Aurelion Wealth Management</h1>", unsafe_allow_html=True)
    st.caption(
        "Diagnostic budgétaire et plan d'investissement personnalisés, propulsés par l'IA (Groq)."
    )
st.divider()

etape1, etape2, etape3 = st.tabs(
    ["Étape 1 — Situation & Documents", "Étape 2 — Profil de risque (optionnel)", "Étape 3 — Diagnostic & Plan"]
)


# ----------------------------------------------------------------------
# ÉTAPE 1 — IDENTITÉ, STATUT PROFESSIONNEL, LOCALISATION, LOGEMENT, ETC.
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
    st.subheader("Statut professionnel")
    statut_professionnel = st.selectbox(
        "Statut professionnel *",
        ["Salarié", "Fonctionnaire", "Retraité", "Demandeur d'emploi", "Entrepreneur / Indépendant"],
    )

    sous_statut = None
    duree_restante_mois = None

    if statut_professionnel == "Salarié":
        sous_statut = st.selectbox(
            "Type de contrat *",
            ["CDI", "CDD", "Intérim", "Stage", "Alternance / Apprentissage"],
        )
        if sous_statut in {"CDD", "Intérim", "Stage", "Alternance / Apprentissage"}:
            duree_restante_mois = st.number_input(
                f"Durée restante du contrat actuel (mois) — {sous_statut} *",
                min_value=0, max_value=60, value=6, step=1,
            )
            st.caption("Cette information affine la cible d'épargne de précaution recommandée (Étape 3).")
    elif statut_professionnel == "Entrepreneur / Indépendant":
        sous_statut = st.selectbox(
            "Forme juridique / statut *",
            ["Auto-entrepreneur / Micro-entreprise", "EI / EIRL", "Société (SASU, EURL, SARL, SAS)", "Profession Libérale / TNS"],
        )
        st.caption(
            "Ce statut influence les conseils retraite/prévoyance et la gestion de trésorerie "
            "recommandés dans le rapport (Étape 3)."
        )
    elif statut_professionnel == "Retraité":
        st.caption("Les recommandations retraite (PER) seront adaptées : priorité à la transmission et à l'assurance-vie.")
    elif statut_professionnel == "Demandeur d'emploi":
        st.caption("Le plan d'investissement priorisera la reconstitution du matelas de sécurité.")

    st.divider()
    st.subheader("Localisation")
    col_loc1, col_loc2 = st.columns(2)
    with col_loc1:
        region = st.selectbox("Région *", REGIONS, index=REGIONS.index("Île-de-France"))
    with col_loc2:
        code_postal = st.text_input("Code postal (optionnel)", max_chars=5, placeholder="ex. 75011")
    if code_postal and not re.fullmatch(r"\d{5}", code_postal):
        st.warning("Le code postal doit contenir 5 chiffres. Il ne sera pas utilisé tel quel.")
        code_postal = ""
    zone_cout_vie = "Élevé" if region in REGIONS_COUT_ELEVE else "Modéré"
    st.caption(f"Zone de coût de la vie estimée pour cette région : {zone_cout_vie}.")

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
        montant_logement = st.number_input("Loyer mensuel (€) *", min_value=0.0, value=0.0, step=10.0, format="%.2f")
    elif statut_logement == "Propriétaire avec crédit immobilier en cours":
        montant_logement = st.number_input(
            "Mensualité du crédit immobilier (€) *", min_value=0.0, value=0.0, step=10.0, format="%.2f"
        )
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
            help=TEXTE_AIDE_TMI,
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
    st.subheader("Salaire net")
    mode_saisie_salaire = st.radio(
        "Comment souhaitez-vous renseigner votre salaire net ? *",
        ["Importer ma fiche de paie (PDF)", "Saisir le montant manuellement"],
        horizontal=True,
    )

    fichier_pdf = None
    salaire_net_manuel = None

    if mode_saisie_salaire == "Importer ma fiche de paie (PDF)":
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
    else:
        salaire_net_manuel = st.number_input(
            "Salaire net mensuel (€) *", min_value=0.0, value=0.0, step=10.0, format="%.2f"
        )

    st.info("Passez ensuite à l'Étape 2 (facultative) ou directement à l'Étape 3.")


# ----------------------------------------------------------------------
# ÉTAPE 2 — PROFIL DE RISQUE (OPTIONNEL)
# ----------------------------------------------------------------------
with etape2:
    st.subheader("Votre rapport au risque (facultatif)")
    st.caption(
        "Ces questions permettent d'affiner l'allocation entre épargne de précaution, "
        "PER et investissement long terme. Vous pouvez laisser ces champs vides : "
        "un profil équilibré par défaut sera utilisé."
    )

    repondre_qcm = st.checkbox("Je souhaite renseigner mon profil de risque")

    reponses_risque = {"horizon": None, "reaction_baisse": None, "preference_repartition": None}
    niveau_connaissance = "Non renseigné"

    if repondre_qcm:
        for cle, question in QUESTIONS_RISQUE.items():
            reponses_risque[cle] = st.radio(
                question["label"],
                options=list(question["options"].keys()),
                index=None,
                key=f"risque_{cle}",
            )

        connaissance_choisie = st.radio(
            "Comment évaluez-vous vos connaissances financières ?",
            options=list(CONNAISSANCES_FINANCIERES.keys()),
            index=None,
            key="connaissances_financieres",
        )
        niveau_connaissance = CONNAISSANCES_FINANCIERES.get(connaissance_choisie, "Non renseigné")

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
        salaire_net = None

        if mode_saisie_salaire == "Saisir le montant manuellement":
            if salaire_net_manuel and salaire_net_manuel > 0:
                salaire_net = salaire_net_manuel
            else:
                st.warning("Merci de saisir un salaire net mensuel valide (Étape 1) avant de continuer.")
        else:
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
            st.success(f"Salaire net retenu : {salaire_net:.2f} €")

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
                zone_cout_vie=zone_cout_vie,
            )

            temps2 = construire_allocation_temps2(
                capacite_epargne=temps1["capacite_epargne"],
                depenses_essentielles=temps1["depenses_essentielles"],
                epargne_disponible=epargne_disponible,
                age=age,
                tmi_pct=tmi_pct,
                profil_risque=profil_risque,
                statut_professionnel=statut_professionnel,
                sous_statut=sous_statut,
                duree_restante_mois=duree_restante_mois,
            )

            identite = {
                "age": age,
                "situation_matrimoniale": situation_matrimoniale,
                "nb_parts_fiscales": nb_parts_fiscales,
                "nb_adultes": nb_adultes,
                "nb_enfants": nb_enfants,
                "handicap": handicap,
            }
            emploi = {
                "statut_professionnel": statut_professionnel,
                "sous_statut": sous_statut,
                "duree_restante_mois": duree_restante_mois,
            }
            localisation = {"region": region, "code_postal": code_postal, "zone_cout_vie": zone_cout_vie}
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
                    emploi=emploi,
                    localisation=localisation,
                    logement=logement,
                    dettes=dettes,
                    patrimoine=patrimoine,
                    temps1=temps1,
                    temps2=temps2,
                    profil_risque=profil_risque,
                    score_risque=score_risque,
                    niveau_connaissance=niveau_connaissance,
                )

            st.session_state.resultat = (temps1, temps2, rapport)
        elif mode_saisie_salaire == "Importer ma fiche de paie (PDF)" and fichier_pdf is not None:
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
        st.subheader("Temps 1 — Reste à vivre réel et répartition du budget")
        st.caption(f"Plaisirs plafonnés à {temps1['taux_plafond_plaisir']*100:.0f}% du revenu total — l'excédent finance l'épargne.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dépenses essentielles", f"{temps1['depenses_essentielles']:.2f} €")
        col2.metric("Impôts & taxes (estimation)", f"{temps1['impots_mensuels']:.2f} €")
        col3.metric("Plaisirs (plafonnés)", f"{temps1['plaisirs']:.2f} €")
        col4.metric("Capacité d'épargne", f"{temps1['capacite_epargne']:.2f} €")

        df_temps1 = pd.DataFrame(
            {
                "Poste": ["Dépenses essentielles", "Impôts & taxes", "Plaisirs (plafonnés)", "Capacité d'épargne"],
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
                    marker_color=["#0A1F44", "#7A8CA8", "#C9A44C", "#2E7D5B"],
                )
            ]
        )
        fig1.update_layout(
            title="Répartition du revenu total mensuel",
            yaxis_title="Montant (€)",
            showlegend=False,
            margin=dict(t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()
        st.subheader("Temps 2 — Plan d'investissement personnalisé")
        st.caption(
            f"Couverture actuelle : {temps2['mois_couverture_actuelle']:.1f} mois de charges essentielles "
            f"(cible recommandée : {temps2['cible_matelas_mois']} mois, ajustée selon le profil de risque "
            "et le statut professionnel)."
        )

        col5, col6, col7 = st.columns(3)
        col5.metric("Épargne de précaution", f"{temps2['montant_matelas']:.2f} €")
        col6.metric("PER (fiscal / retraite)", f"{temps2['montant_per']:.2f} €")
        col7.metric("Investissement long terme", f"{temps2['montant_invest_long_terme']:.2f} €")

        df_temps2 = pd.DataFrame(
            {
                "Allocation": ["Épargne de précaution", "PER", "Investissement long terme"],
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
                    marker_color=["#0A1F44", "#C9A44C", "#2E7D5B"],
                )
            ]
        )
        fig2.update_layout(
            title="Allocation de la capacité d'épargne mensuelle",
            yaxis_title="Montant (€)",
            showlegend=False,
            margin=dict(t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
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
            "investissement personnalisé au sens réglementaire. Les repères de marché mentionnés "
            "proviennent des connaissances générales du modèle d'IA, pas d'une donnée de marché "
            "en temps réel : vérifiez les taux et conditions actuels avant toute décision, et "
            "consultez un conseiller en gestion de patrimoine agréé."
        )
    elif not lancer:
        st.info("Complétez l'Étape 1 (et éventuellement l'Étape 2), puis cliquez sur \"Lancer l'analyse IA\".")
