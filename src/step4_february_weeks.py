import joblib

from step1_load_data import load_data, NUM_FEATURES, CAT_FEATURES, TARGET
from step3_production_model import add_predictions

FEATURES = NUM_FEATURES + CAT_FEATURES

WEEK_WINDOWS = {
    "semaine1": ("2011-02-01 00:00:00", "2011-02-07 23:00:00"),
    "semaine2": ("2011-02-08 00:00:00", "2011-02-14 23:00:00"),
    "semaine3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}


def get_week(df, start, end):
    """
    Filtre le dataset complet sur une fenetre temporelle donnee.
    """
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    return df.loc[mask].copy()


if __name__ == "__main__":
    df = load_data()
    model = joblib.load("model_production.joblib")

    weeks_with_pred = {}

    for week_name, (start, end) in WEEK_WINDOWS.items():
        week_data = get_week(df, start, end)
        week_data_with_pred = add_predictions(model, week_data)
        weeks_with_pred[week_name] = week_data_with_pred

        # Sauvegarde pour reutilisation dans les etapes suivantes
        week_data_with_pred.to_parquet(f"{week_name}_with_pred.parquet")

        print(f"{week_name} : {week_data_with_pred.shape[0]} lignes"
              f" ({week_data_with_pred['datetime'].min()} -> {week_data_with_pred['datetime'].max()})")

    print("\nLes 3 semaines ont ete sauvegardees avec leurs predictions "
          "(semaine1_with_pred.parquet, semaine2_with_pred.parquet, semaine3_with_pred.parquet)")
