import pandas as pd

df = pd.read_csv("data/raw/JFF_formatted_public_release_values.csv", encoding="latin1")

FEATURES = []

df_features = df[FEATURES]

df_features.to_parquet("gallup_selected_features.parquet", index=False)