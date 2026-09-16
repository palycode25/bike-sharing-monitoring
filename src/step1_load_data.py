import pandas as pd

# Colonnes du dataset, telles que définies par l'énoncé
NUM_FEATURES = ["temp", "atemp", "hum", "windspeed"]
CAT_FEATURES = ["season", "holiday", "workingday", "mnth", "hr", "weekday"]
TARGET = "cnt"


def load_data(path="data/hour.csv"):
    """
    Charge le dataset Bike Sharing et reconstruit un timestamp horaire
    complet a partir de dteday (date) + hr (heure).
    """
    df = pd.read_csv(path)

    # dteday est une date sans heure -> on lui ajoute hr pour avoir un vrai datetime
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    return df


if __name__ == "__main__":
    df = load_data()

    print("Shape :", df.shape)
    print("\nPeriode couverte :", df["datetime"].min(), "->", df["datetime"].max())
    print("\nColonnes numeriques :", NUM_FEATURES)
    print("Colonnes categorielles :", CAT_FEATURES)
    print("Cible :", TARGET)
    print("\nApercu :")
    print(df[["datetime"] + NUM_FEATURES + CAT_FEATURES + [TARGET]].head())
