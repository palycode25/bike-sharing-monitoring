import joblib
from sklearn.linear_model import LinearRegression

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import RegressionPreset

from step1_load_data import load_data, NUM_FEATURES, CAT_FEATURES, TARGET
from step2_reference_and_validation import get_january_reference

FEATURES = NUM_FEATURES + CAT_FEATURES


def train_production_model(jan_data):
    """
    Entraine le modele de production sur la totalite de janvier (sans split).
    Ce modele servira de reference pour comparer les semaines de fevrier.
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


def generate_production_report(jan_data_with_pred, column_mapping):
    """
    Rapport de production : baseline du modele evalue sur janvier lui-meme.
    """
    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=jan_data_with_pred,
        current_data=jan_data_with_pred,
        column_mapping=column_mapping,
    )
    return report


if __name__ == "__main__":
    df = load_data()
    jan_data = get_january_reference(df)

    model = train_production_model(jan_data)
    print("Modele de production entraine sur", jan_data.shape[0], "lignes de janvier.")

    jan_data_with_pred = add_predictions(model, jan_data)

    column_mapping = ColumnMapping(
        target=TARGET,
        prediction="prediction",
        numerical_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES,
    )

    production_report = generate_production_report(jan_data_with_pred, column_mapping)
    production_report.save_html("production_report_janvier.html")
    print("Rapport de production sauvegarde : production_report_janvier.html")

    # Sauvegarde du modele et des donnees de reference pour les etapes suivantes
    joblib.dump(model, "model_production.joblib")
    jan_data_with_pred.to_parquet("jan_data_with_pred.parquet")
    print("Modele et donnees de reference sauvegardes (model_production.joblib, jan_data_with_pred.parquet)")
