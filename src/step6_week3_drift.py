import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from step1_load_data import NUM_FEATURES, CAT_FEATURES, TARGET


def generate_global_drift_report(reference_data, current_data, column_mapping):
    """
    Rapport de derive globale : toutes les variables (numeriques + categorielles + prediction).
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    return report


def generate_numerical_drift_report(reference_data, current_data, column_mapping_num_only):
    """
    Rapport de derive restreint aux seules variables numeriques.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping_num_only,
    )
    return report


if __name__ == "__main__":
    jan_data = pd.read_parquet("jan_data_with_pred.parquet")
    semaine3 = pd.read_parquet("semaine3_with_pred.parquet")

    # --- Rapport 1 : derive globale (toutes les variables) ---
    column_mapping_full = ColumnMapping(
        target=TARGET,
        prediction="prediction",
        numerical_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES,
    )

    global_drift_report = generate_global_drift_report(jan_data, semaine3, column_mapping_full)
    global_drift_report.save_html("week3_global_drift_report.html")
    print("Rapport de derive globale (semaine 3) sauvegarde : week3_global_drift_report.html")

    # --- Rapport 2 : derive numerique seule (temp, atemp, hum, windspeed) ---
    column_mapping_num_only = ColumnMapping(
        numerical_features=NUM_FEATURES,
        categorical_features=[],
    )

    jan_data_num = jan_data[NUM_FEATURES]
    semaine3_num = semaine3[NUM_FEATURES]

    numerical_drift_report = generate_numerical_drift_report(
        jan_data_num, semaine3_num, column_mapping_num_only
    )
    numerical_drift_report.save_html("week3_numerical_drift_report.html")
    print("Rapport de derive numerique seule (semaine 3) sauvegarde : week3_numerical_drift_report.html")
