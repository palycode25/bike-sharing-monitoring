import pandas as pd
import joblib

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import RegressionPreset, TargetDriftPreset, DataDriftPreset
from evidently.ui.workspace import Workspace

from step1_load_data import NUM_FEATURES, CAT_FEATURES, TARGET
from step2_reference_and_validation import train_validation_model
from step5_target_drift import compute_mae

WORKSPACE_NAME = "bike_monitoring_workspace"
PROJECT_NAME = "bike_sharing_monitoring"
PROJECT_DESCRIPTION = "Examen - Cas d'etude workflow de monitoring (UCI Bike Sharing)"


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


if __name__ == "__main__":
    # --- Chargement des donnees intermediaires deja calculees ---
    jan_data = pd.read_parquet("jan_data_with_pred.parquet")
    semaine1 = pd.read_parquet("semaine1_with_pred.parquet")
    semaine2 = pd.read_parquet("semaine2_with_pred.parquet")
    semaine3 = pd.read_parquet("semaine3_with_pred.parquet")

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

    # --- 1. Rapport de validation (train/test sur janvier) ---
    # on reconstruit train/test a partir de jan_data (sans predictions) pour rester coherent
    _, train_data, test_data = train_validation_model(jan_data.drop(columns=["prediction"]))
    validation_report = Report(metrics=[RegressionPreset()])
    validation_report.run(
        reference_data=train_data, current_data=test_data, column_mapping=column_mapping_full
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, validation_report)
    print("1. Rapport de validation ajoute au workspace")

    # --- 2. Rapport de production janvier (janvier vs janvier) ---
    production_report = Report(metrics=[RegressionPreset()])
    production_report.run(
        reference_data=jan_data, current_data=jan_data, column_mapping=column_mapping_full
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, production_report)
    print("2. Rapport de production janvier ajoute au workspace")

    # --- 3. Rapport de derive cible (pire semaine entre S1 et S2) ---
    mae_s1 = compute_mae(semaine1)
    mae_s2 = compute_mae(semaine2)
    worst_week_data = semaine1 if mae_s1 >= mae_s2 else semaine2

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=jan_data, current_data=worst_week_data, column_mapping=column_mapping_full
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, target_drift_report)
    print("3. Rapport de derive cible ajoute au workspace")

    # --- 4. Rapport de derive globale semaine 3 ---
    global_drift_report = Report(metrics=[DataDriftPreset()])
    global_drift_report.run(
        reference_data=jan_data, current_data=semaine3, column_mapping=column_mapping_full
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, global_drift_report)
    print("4. Rapport de derive globale (semaine 3) ajoute au workspace")

    # --- 5. Rapport de derive numerique seule semaine 3 ---
    numerical_drift_report = Report(metrics=[DataDriftPreset()])
    numerical_drift_report.run(
        reference_data=jan_data[NUM_FEATURES],
        current_data=semaine3[NUM_FEATURES],
        column_mapping=column_mapping_num_only,
    )
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, numerical_drift_report)
    print("5. Rapport de derive numerique seule (semaine 3) ajoute au workspace")

    print(f"\nTous les rapports ont ete ajoutes au projet '{PROJECT_NAME}' "
          f"dans le workspace '{WORKSPACE_NAME}'.")
