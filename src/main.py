import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import RegressionPreset, TargetDriftPreset, DataDriftPreset
from evidently.ui.workspace import Workspace


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DATA_PATH = "data/hour.csv"

NUM_FEATURES = ["temp", "atemp", "hum", "windspeed"]
CAT_FEATURES = ["season", "holiday", "workingday", "mnth", "hr", "weekday"]
TARGET = "cnt"
FEATURES = NUM_FEATURES + CAT_FEATURES

JAN_START = "2011-01-01 00:00:00"
JAN_END = "2011-01-31 23:00:00"

WEEK_WINDOWS = {
    "semaine1": ("2011-02-01 00:00:00", "2011-02-07 23:00:00"),
    "semaine2": ("2011-02-08 00:00:00", "2011-02-14 23:00:00"),
    "semaine3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}

WORKSPACE_NAME = "bike_monitoring_workspace"
PROJECT_NAME = "bike_sharing_monitoring"
PROJECT_DESCRIPTION = "Examen - Cas d'etude workflow de monitoring (UCI Bike Sharing)"


# ---------------------------------------------------------------------------
# Chargement des donnees
# ---------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """
    Charge le dataset Bike Sharing et reconstruit un timestamp horaire complet
    a partir de dteday (date) + hr (heure).
    """
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df


def filter_period(df, start, end):
    """
    Filtre le dataset sur une fenetre temporelle [start, end] incluse.
    """
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Modeles
# ---------------------------------------------------------------------------
def train_validation_model(jan_data):
    """
    Split train/test sur janvier, entraine un modele de regression,
    ajoute les predictions sur les deux sous-ensembles.
    """
    train_data, test_data = train_test_split(jan_data, test_size=0.3, random_state=42)
    train_data = train_data.copy()
    test_data = test_data.copy()

    model = LinearRegression()
    model.fit(train_data[FEATURES], train_data[TARGET])

    train_data["prediction"] = model.predict(train_data[FEATURES])
    test_data["prediction"] = model.predict(test_data[FEATURES])

    return train_data, test_data


def train_production_model(jan_data):
    """
    Entraine le modele de production sur la totalite de janvier (sans split).
    """
    model = LinearRegression()
    model.fit(jan_data[FEATURES], jan_data[TARGET])
    return model


def add_predictions(model, df):
    """
    Ajoute une colonne 'prediction' a un DataFrame en utilisant le modele donne.
    """
    df = df.copy()
    df["prediction"] = model.predict(df[FEATURES])
    return df


def compute_mae(df):
    """
    Calcule le MAE entre la cible reelle et la prediction du modele.
    """
    return mean_absolute_error(df[TARGET], df["prediction"])


# ---------------------------------------------------------------------------
# Workspace Evidently
# ---------------------------------------------------------------------------
def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Ajoute un rapport a un projet existant ou nouvellement cree dans le workspace.
    """
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description
        project.save()

    workspace.add_report(project.id, report)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("=== 1. Chargement des donnees ===")
    df = load_data()
    print(f"Dataset complet : {df.shape[0]} lignes ({df['datetime'].min()} -> {df['datetime'].max()})")

    print("\n=== 2. Construction du dataset de reference (janvier 2011) ===")
    jan_data = filter_period(df, JAN_START, JAN_END)
    print(f"Janvier 2011 : {jan_data.shape[0]} lignes")

    column_mapping_full = ColumnMapping(
        target=TARGET,
        prediction="prediction",
        numerical_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES,
    )
    column_mapping_num_only = ColumnMapping(
        numerical_features=NUM_FEATURES,
        categorical_features=[],
    )

    workspace = Workspace.create(WORKSPACE_NAME)

    print("\n=== 3. Split train/test sur janvier + rapport de validation ===")
    train_data, test_data = train_validation_model(jan_data)
    print(f"Train : {train_data.shape[0]} lignes | Test : {test_data.shape[0]} lignes")

    validation_report = Report(metrics=[RegressionPreset()])
    validation_report.run(
        reference_data=train_data, current_data=test_data, column_mapping=column_mapping_full
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, validation_report)
    print("Rapport de validation (RegressionPreset) ajoute au workspace")

    print("\n=== 4. Modele de production sur janvier entier + rapport de production ===")
    production_model = train_production_model(jan_data)
    jan_data_with_pred = add_predictions(production_model, jan_data)

    production_report = Report(metrics=[RegressionPreset()])
    production_report.run(
        reference_data=jan_data_with_pred, current_data=jan_data_with_pred,
        column_mapping=column_mapping_full,
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, production_report)
    print("Rapport de production janvier (RegressionPreset) ajoute au workspace")

    print("\n=== 5. Predictions sur les 3 semaines de fevrier ===")
    weeks_with_pred = {}
    for week_name, (start, end) in WEEK_WINDOWS.items():
        week_data = filter_period(df, start, end)
        weeks_with_pred[week_name] = add_predictions(production_model, week_data)
        print(f"{week_name} : {weeks_with_pred[week_name].shape[0]} lignes ({start} -> {end})")

    print("\n=== 6. Comparaison MAE semaine 1 vs semaine 2 + rapport de derive cible ===")
    mae_s1 = compute_mae(weeks_with_pred["semaine1"])
    mae_s2 = compute_mae(weeks_with_pred["semaine2"])
    print(f"MAE semaine 1 : {mae_s1:.3f}")
    print(f"MAE semaine 2 : {mae_s2:.3f}")

    if mae_s1 >= mae_s2:
        worst_week_name, worst_week_data = "semaine1", weeks_with_pred["semaine1"]
    else:
        worst_week_name, worst_week_data = "semaine2", weeks_with_pred["semaine2"]
    print(f"Pire semaine retenue : {worst_week_name} (MAE = {max(mae_s1, mae_s2):.3f})")

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=jan_data_with_pred, current_data=worst_week_data,
        column_mapping=column_mapping_full,
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, target_drift_report)
    print("Rapport de derive cible (TargetDriftPreset) ajoute au workspace")

    print("\n=== 7. Semaine 3 : derive globale + derive numerique seule ===")
    semaine3 = weeks_with_pred["semaine3"]

    global_drift_report = Report(metrics=[DataDriftPreset()])
    global_drift_report.run(
        reference_data=jan_data_with_pred, current_data=semaine3,
        column_mapping=column_mapping_full,
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, global_drift_report)
    print("Rapport de derive globale semaine 3 (DataDriftPreset) ajoute au workspace")

    numerical_drift_report = Report(metrics=[DataDriftPreset()])
    numerical_drift_report.run(
        reference_data=jan_data_with_pred[NUM_FEATURES],
        current_data=semaine3[NUM_FEATURES],
        column_mapping=column_mapping_num_only,
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, numerical_drift_report)
    print("Rapport de derive numerique seule semaine 3 (DataDriftPreset) ajoute au workspace")

    print(f"\n=== Termine : 5 rapports ajoutes au projet '{PROJECT_NAME}' "
          f"dans le workspace '{WORKSPACE_NAME}' ===")
