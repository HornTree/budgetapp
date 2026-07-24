"""
Application Streamlit — Générateur de Budget Personnalisé
-----------------------------------------------------------
Cette application :
1. Récupère la situation du foyer via un questionnaire.
2. Extrait le texte d'une fiche de paie PDF (pdfplumber).
3. Interroge un modèle Mistral local via Ollama pour extraire le salaire net.
4. Calcule une répartition budgétaire inspirée de la règle 50/30/20, adaptée
   au contexte du foyer (handicap, taille du foyer, dettes).
5. Demande à Ollama de rédiger un rapport Markdown structuré et bienveillant.

Prérequis :
- Ollama installé et lancé en local (`ollama serve`)
- Modèle mistral disponible (`ollama pull mistral`)
- pip install streamlit pdfplumber requests
"""

import json
import re

import pdfplumber
import requests
import streamlit as st

# ----------------------------------------------------------------------
# Configuration générale
# ----------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

st.set_page_config(page_title="Mon Budget Personnalisé", page_icon="💶", layout="centered")


# ----------------------------------------------------------------------
# 1. EXTRACTION DU TEXTE DU PDF
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
# 2. APPEL GÉNÉRIQUE À L'API OLLAMA
# ----------------------------------------------------------------------
def appeler_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Envoie un prompt au modèle local via l'API Ollama (/api/generate)
    et retourne la réponse texte générée (champ "response").

    Utilise stream=False pour récupérer la réponse complète en un seul appel.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        reponse = requests.post(OLLAMA_URL, json=payload, timeout=180)
        reponse.raise_for_status()
        data = reponse.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        st.error(
            "Impossible de contacter Ollama sur http://localhost:11434. "
            "Vérifiez qu'Ollama est bien lancé (`ollama serve`) et que le "
            "modèle 'mistral' est installé (`ollama pull mistral`)."
        )
        return ""
    except Exception as e:
        st.error(f"Erreur lors de l'appel à Ollama : {e}")
        return ""


# ----------------------------------------------------------------------
# 3. EXTRACTION DU SALAIRE NET VIA L'IA (format JSON)
# ----------------------------------------------------------------------
def extraire_salaire_net(texte_paie: str) -> float | None:
    """
    Construit un prompt demandant à Mistral d'extraire le "Salaire Net à
    payer" du texte de la fiche de paie et de le renvoyer en JSON strict.
    Retourne le montant en float, ou None si l'extraction échoue.
    """
    prompt = f"""Tu es un assistant spécialisé dans la lecture de fiches de paie françaises.
Voici le texte brut extrait d'une fiche de paie :

---
{texte_paie}
---

Ta tâche : trouve le montant du "Salaire Net à payer" (parfois appelé
"Net à payer", "Salaire net", "Net payé").

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou
après, au format exact suivant :
{{"salaire_net": 0000.00}}

Si tu ne trouves aucun montant, réponds avec :
{{"salaire_net": null}}
"""
    reponse = appeler_ollama(prompt)
    if not reponse:
        return None

    # On extrait le premier bloc JSON présent dans la réponse (au cas où
    # le modèle ajoute du texte autour malgré la consigne).
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
# 4. MOTEUR DE CALCUL — Logique 50/30/20 adaptée
# ----------------------------------------------------------------------
def calculer_repartition(
    salaire_net: float,
    nb_adultes: int,
    nb_enfants: int,
    handicap: bool,
    montant_dette: float,
) -> dict:
    """
    Calcule la répartition du salaire net selon une règle 50/30/20 adaptée :

    - Charges fixes (Besoins) : 50% du salaire net.
      -> passe à 60% si handicap/frais médicaux importants OU si le foyer
         compte plus de 3 personnes (adultes + enfants).
    - Dettes / Épargne : 20% du salaire net.
      -> si des dettes existent (montant_dette > 0), cette part est
         prioritairement affectée au remboursement de la dette.
    - Loisirs / Plaisir : le reste (le solde après les deux postes ci-dessus).

    Retourne un dictionnaire avec les montants en euros et les taux appliqués.
    """
    nb_personnes = nb_adultes + nb_enfants

    # Détermination du taux "Besoins"
    taux_besoins = 0.60 if (handicap or nb_personnes > 3) else 0.50
    taux_dette_epargne = 0.20
    # Le reste va aux loisirs
    taux_loisirs = 1.0 - taux_besoins - taux_dette_epargne

    montant_besoins = round(salaire_net * taux_besoins, 2)
    montant_dette_epargne = round(salaire_net * taux_dette_epargne, 2)
    montant_loisirs = round(salaire_net * taux_loisirs, 2)

    # Si le foyer a des dettes, on précise que ce poste sert en priorité
    # à rembourser la dette (au lieu d'épargner).
    a_des_dettes = montant_dette > 0

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
    }


# ----------------------------------------------------------------------
# 5. GÉNÉRATION DU RAPPORT MARKDOWN VIA L'IA
# ----------------------------------------------------------------------
def generer_rapport(
    repartition: dict,
    nb_adultes: int,
    nb_enfants: int,
    handicap: bool,
    montant_dette: float,
    mensualite_dette: float,
) -> str:
    """
    Construit un prompt détaillé décrivant la situation du foyer et les
    montants calculés, puis demande à Mistral de rédiger un rapport
    Markdown structuré (diagnostic, tableau de répartition, conseils).
    """
    prompt = f"""Tu es un conseiller budgétaire familial, bienveillant et pédagogue.
Rédige un rapport en Markdown clair et structuré à partir des données suivantes.

DONNÉES DU FOYER :
- Nombre d'adultes : {nb_adultes}
- Nombre d'enfants : {nb_enfants}
- Handicap / frais médicaux importants : {"Oui" if handicap else "Non"}
- Montant total des dettes : {montant_dette:.2f} €
- Mensualité de remboursement des dettes : {mensualite_dette:.2f} €
- Salaire net mensuel : {repartition['salaire_net']:.2f} €

RÉPARTITION BUDGÉTAIRE CALCULÉE :
- Charges fixes / Besoins ({repartition['taux_besoins']*100:.0f}%) : {repartition['montant_besoins']:.2f} €
- Dettes / Épargne ({repartition['taux_dette_epargne']*100:.0f}%) : {repartition['montant_dette_epargne']:.2f} €
- Loisirs / Plaisir ({repartition['taux_loisirs']*100:.0f}%) : {repartition['montant_loisirs']:.2f} €

CONSIGNES DE RÉDACTION :
Structure le rapport en exactement 3 parties, avec des titres Markdown (##) :

1. **Diagnostic rapide du foyer** : un court paragraphe résumant la
   situation (taille du foyer, présence de dettes, contexte médical
   éventuel) et le niveau de tension budgétaire.

2. **Répartition budgétaire recommandée** : présente un tableau Markdown
   (colonnes : Poste | Pourcentage | Montant en €) reprenant les 3 postes
   ci-dessus, avec une phrase d'explication sur pourquoi ces pourcentages
   ont été retenus pour ce foyer.

3. **3 conseils pratiques et bienveillants** : liste numérotée de 3
   conseils concrets, adaptés à la situation réelle du foyer (dettes,
   handicap, taille du foyer), sans jugement, avec un ton encourageant.

Réponds uniquement avec le rapport en Markdown, sans phrase d'introduction
avant le titre.
"""
    return appeler_ollama(prompt)


# ----------------------------------------------------------------------
# INTERFACE UTILISATEUR STREAMLIT
# ----------------------------------------------------------------------
st.title("💶 Mon Budget Personnalisé")
st.caption(
    "Renseignez votre situation, importez votre fiche de paie, et obtenez "
    "une répartition budgétaire générée par IA (Mistral via Ollama, en local)."
)

# --- Questionnaire dans la sidebar ---
st.sidebar.header("📋 Votre situation")

nb_adultes = st.sidebar.number_input("Nombre d'adultes dans le foyer", min_value=1, max_value=10, value=1, step=1)
nb_enfants = st.sidebar.number_input("Nombre d'enfants dans le foyer", min_value=0, max_value=10, value=0, step=1)

handicap = st.sidebar.checkbox("Handicap / Frais médicaux importants")

montant_dette = st.sidebar.number_input(
    "Montant total des dettes (€)", min_value=0.0, value=0.0, step=100.0, format="%.2f"
)
mensualite_dette = st.sidebar.number_input(
    "Mensualité de remboursement des dettes (€)", min_value=0.0, value=0.0, step=10.0, format="%.2f"
)

st.sidebar.header("📄 Fiche de paie")
fichier_pdf = st.sidebar.file_uploader("Importer votre fiche de paie (PDF)", type=["pdf"])

lancer = st.sidebar.button("🚀 Générer mon budget", use_container_width=True)


# ----------------------------------------------------------------------
# LOGIQUE PRINCIPALE
# ----------------------------------------------------------------------
if lancer:
    if fichier_pdf is None:
        st.warning("Merci d'importer votre fiche de paie au format PDF avant de continuer.")
    else:
        # Étape 1 : extraction du texte du PDF
        with st.spinner("Lecture du PDF en cours..."):
            texte_pdf = extraire_texte_pdf(fichier_pdf)

        if not texte_pdf:
            st.error("Le texte n'a pas pu être extrait du PDF. Le fichier est peut-être scanné (image) ou vide.")
        else:
            with st.expander("Voir le texte extrait de la fiche de paie"):
                st.text(texte_pdf)

            # Étape 2 : extraction du salaire net via l'IA
            with st.spinner("Extraction du salaire net via l'IA (Mistral)..."):
                salaire_net = extraire_salaire_net(texte_pdf)

            if salaire_net is None:
                st.error(
                    "Le salaire net n'a pas pu être détecté automatiquement. "
                    "Vérifiez que la fiche de paie contient bien la mention "
                    "'Net à payer' ou renseignez-le manuellement ci-dessous."
                )
                salaire_net = st.number_input(
                    "Salaire net mensuel (€) — saisie manuelle", min_value=0.0, value=0.0, step=10.0
                )

            if salaire_net and salaire_net > 0:
                st.success(f"Salaire net détecté : **{salaire_net:.2f} €**")

                # Étape 3 : calcul de la répartition budgétaire
                repartition = calculer_repartition(
                    salaire_net=salaire_net,
                    nb_adultes=nb_adultes,
                    nb_enfants=nb_enfants,
                    handicap=handicap,
                    montant_dette=montant_dette,
                )

                # Affichage rapide des métriques
                col1, col2, col3 = st.columns(3)
                col1.metric("Besoins", f"{repartition['montant_besoins']:.2f} €", f"{repartition['taux_besoins']*100:.0f}%")
                col2.metric(
                    "Dettes / Épargne",
                    f"{repartition['montant_dette_epargne']:.2f} €",
                    f"{repartition['taux_dette_epargne']*100:.0f}%",
                )
                col3.metric("Loisirs", f"{repartition['montant_loisirs']:.2f} €", f"{repartition['taux_loisirs']*100:.0f}%")

                # Étape 4 : génération du rapport via l'IA
                with st.spinner("Rédaction du rapport personnalisé via l'IA (Mistral)..."):
                    rapport = generer_rapport(
                        repartition=repartition,
                        nb_adultes=nb_adultes,
                        nb_enfants=nb_enfants,
                        handicap=handicap,
                        montant_dette=montant_dette,
                        mensualite_dette=mensualite_dette,
                    )

                st.divider()
                if rapport:
                    st.markdown(rapport)
                else:
                    st.error("Le rapport n'a pas pu être généré. Vérifiez qu'Ollama est bien lancé.")
            else:
                st.info("Renseignez un salaire net valide pour lancer le calcul du budget.")
else:
    st.info("Renseignez votre situation dans le menu latéral, importez votre fiche de paie, puis cliquez sur **Générer mon budget**.")
