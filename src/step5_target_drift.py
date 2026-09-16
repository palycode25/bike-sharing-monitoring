import pandas as pd
from sklearn.metrics import mean_absolute_error

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset

from step1_load_data import NUM_FEATURES, CAT_FEATURES, TARGET


def compute_mae(df):
    """
    Calcule le MAE entre la cible reelle et la prediction du modele.
    """
    return mean_absolute_error(df[TARGET], df["prediction"])


def generate_target_drift_report(reference_data, current_data, column_mapping):
    """
    Rapport de derive cible : compare la reference (janvier) a la semaine
    la plus degradee (celle avec le plus grand MAE entre semaine1 et semaine2).
    """
    report = Report(metrics=[TargetDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    return report


if __name__ == "__main__":
    jan_data = pd.read_parquet("jan_data_with_pred.parquet")
    semaine1 = pd.read_parquet("semaine1_with_pred.parquet")
    semaine2 = pd.read_parquet("semaine2_with_pred.parquet")

    mae_s1 = compute_mae(semaine1)
    mae_s2 = compute_mae(semaine2)

    print(f"MAE semaine 1 : {mae_s1:.3f}")
    print(f"MAE semaine 2 : {mae_s2:.3f}")

    if mae_s1 >= mae_s2:
        worst_week_name = "semaine1"
        worst_week_data = semaine1
    else:
        worst_week_name = "semaine2"
        worst_week_data = semaine2

    print(f"\nPire semaine retenue : {worst_week_name} (MAE = {max(mae_s1, mae_s2):.3f})")

    column_mapping = ColumnMapping(
        target=TARGET,
        prediction="prediction",
        numerical_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES,
    )

    target_drift_report = generate_target_drift_report(jan_data, worst_week_data, column_mapping)
    target_drift_report.save_html("target_drift_report.html")
    print("Rapport de derive cible sauvegarde : target_drift_report.html")
