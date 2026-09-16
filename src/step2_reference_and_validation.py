import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import RegressionPreset

from step1_load_data import load_data, NUM_FEATURES, CAT_FEATURES, TARGET

JAN_START = "2011-01-01 00:00:00"
JAN_END = "2011-01-31 23:00:00"


def get_january_reference(df):
    """
    Filtre le dataset complet sur la periode de reference : janvier 2011 entier.
    """
    mask = (df["datetime"] >= JAN_START) & (df["datetime"] <= JAN_END)
    jan_data = df.loc[mask].copy()
    return jan_data


def train_validation_model(jan_data):
    """
    Split train/test sur janvier, entraine un modele de regression,
    ajoute les predictions sur les deux sous-ensembles.
    """
    features = NUM_FEATURES + CAT_FEATURES

    train_data, test_data = train_test_split(
        jan_data, test_size=0.3, random_state=42
    )
    train_data = train_data.copy()
    test_data = test_data.copy()

    model = LinearRegression()
    model.fit(train_data[features], train_data[TARGET])

    train_data["prediction"] = model.predict(train_data[features])
    test_data["prediction"] = model.predict(test_data[features])

    return model, train_data, test_data


def generate_validation_report(train_data, test_data, column_mapping):
    """
    Rapport de validation : compare train (reference) vs test (current)
    avec RegressionPreset.
    """
    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=train_data,
        current_data=test_data,
        column_mapping=column_mapping,
    )
    return report


if __name__ == "__main__":
    df = load_data()
    jan_data = get_january_reference(df)

    print("Nombre de lignes en janvier 2011 :", jan_data.shape[0])
    print("Premiere date :", jan_data["datetime"].min())
    print("Derniere date :", jan_data["datetime"].max())

    column_mapping = ColumnMapping(
        target=TARGET,
        prediction="prediction",
        numerical_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES,
    )

    model, train_data, test_data = train_validation_model(jan_data)
    print("\nTaille train :", train_data.shape[0], "| Taille test :", test_data.shape[0])

    validation_report = generate_validation_report(train_data, test_data, column_mapping)
    validation_report.save_html("validation_report.html")
    print("\nRapport de validation sauvegarde : validation_report.html")
