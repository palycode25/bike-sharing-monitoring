cd C:\Users\fatou\Downloads\bike_monitoring
@"
# Bike Sharing Monitoring — Evidently AI

Workflow de monitoring Machine Learning avec **Evidently 0.6.7**, appliqué au dataset [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

## Objectif

Détecter la dérive d'un modèle de régression prédisant le nombre de locations de vélos (`cnt`), en comparant une période de référence (janvier 2011) à trois semaines de février 2011.

## Données

- **Référence** : 2011-01-01 → 2011-01-31 (mois complet)
- **Monitoring** : 3 semaines consécutives de février 2011
  - Semaine 1 : 01/02 → 07/02
  - Semaine 2 : 08/02 → 14/02
  - Semaine 3 : 15/02 → 21/02

**Variables numériques** : ``temp``, ``atemp``, ``hum``, ``windspeed``
**Variables catégorielles** : ``season``, ``holiday``, ``workingday``, ``mnth``, ``hr``, ``weekday``
**Cible** : ``cnt``

## Pipeline (``src/main.py``)

1. Chargement des données et reconstruction du timestamp horaire
2. Construction du dataset de référence (janvier 2011)
3. Split train/test sur janvier + rapport ``RegressionPreset`` (validation)
4. Entraînement du modèle de production sur janvier entier + rapport ``RegressionPreset`` (production)
5. Génération des prédictions sur les 3 semaines de février
6. Comparaison du MAE semaine 1 vs semaine 2 → sélection de la pire semaine → rapport ``TargetDriftPreset``
7. Semaine 3 : rapport ``DataDriftPreset`` global + rapport ``DataDriftPreset`` restreint aux variables numériques
8. Centralisation des 5 rapports dans un unique projet Evidently

## Utilisation

``````bash
python3 -m venv .venv
source .venv/bin/activate
pip install evidently==0.6.7 pandas scikit-learn

python src/main.py

evidently ui --workspace ./bike_monitoring_workspace --port 8001 --host 0.0.0.0
``````

Puis ouvrir ``http://<ip>:8001`` et consulter le projet **bike_sharing_monitoring**, onglet **Reports**.
