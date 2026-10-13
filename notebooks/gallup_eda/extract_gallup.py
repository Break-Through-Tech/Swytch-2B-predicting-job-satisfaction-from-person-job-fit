
import pandas as pd
import os

#Load Gallup data
df = pd.read_csv(
    "data/raw/JFF_formatted_public_release_values.csv",
    encoding="latin1",
    low_memory=False
)

#Select features
features = [
    "Q6", "Q7", "Q8", "Q9", "Q14", "Q16",
    "Q17", "Q20", "Q23", "Q24", "Q27",
    "Q40", "Q28", "Q31A", "Q31B", "Q31C",
    "Q32A", "Q32B", "Q32C",
    "Q51", "REGION", "WORKER_TYPE"
]

target = "Q4"

#Extract selected features and target
gallup = df[features + [target]].copy()

print("Original shape:", gallup.shape)

#Replace special missing-value codes
gallup = gallup.replace(-98, float("nan"))

#Convert columns to numeric
for col in gallup.columns:
    gallup[col] = pd.to_numeric(
        gallup[col], errors="coerce"
    )

#Remove rows with missing target
gallup = gallup.dropna(subset=[target]).copy()

#Check duplicate respondent IDs
if "ENTITY_ID" in df.columns:
    ids = df.loc[gallup.index, "ENTITY_ID"]
    print("Duplicate respondent IDs:", ids.duplicated().sum())

#Check exact duplicate rows (do not automatically delete)
print("Duplicate feature rows:", gallup.duplicated().sum())

#Check target range
invalid_target = ~gallup[target].between(0, 10)
print("Invalid target values:", invalid_target.sum())

#Remove invalid target values
gallup = gallup.loc[~invalid_target].copy()

#Check missing values
print("\nMissing values:")
print(gallup.isna().sum())

#Save cleaned dataset before imputation
os.makedirs("data/interim", exist_ok=True)

gallup.to_csv(
    "data/interim/gallup_cleaned.csv",
    index=False
)

#Create a separate encoded version
encoded = gallup.copy()

#Numerical features
numeric_features = ["Q20", "Q51"]

#Nominal categorical features
nominal_features = [
    "Q7", "Q9", "Q16",
    "REGION", "WORKER_TYPE"
]

#Ordinal features retain their numeric codes
ordinal_features = [
    "Q8", "Q14", "Q17", "Q23", "Q24",
    "Q27", "Q40", "Q28",
    "Q31A", "Q31B", "Q31C",
    "Q32A", "Q32B", "Q32C"
]

#Fill missing numerical and ordinal values
for col in numeric_features + ordinal_features:
    encoded[col] = encoded[col].fillna(
        encoded[col].median()
    )

#Fill missing nominal values
for col in nominal_features:
    encoded[col] = encoded[col].fillna(
        encoded[col].mode().iloc[0]
    )

#One-hot encode nominal features
encoded = pd.get_dummies(
    encoded,
    columns=nominal_features,
    dtype=int
)


#Save encoded dataset
encoded.to_csv(
    "data/interim/gallup_encoded.csv",
    index=False
)

#Print messages
print("\nCleaned shape:", gallup.shape)
print("Encoded shape:", encoded.shape)

print("\nMissing values in encoded data:")
print(encoded.isna().sum()[encoded.isna().sum() > 0])

print("\nFiles saved successfully!")



# Cleaning summary:
# - Selected 22 features and Q4 (job satisfaction) as target
# - Removed 48 respondents with missing target values
# - Removed Q52 and Q57 because of too much missing data
# - Converted -98 response codes to NaN
# - Checked for duplicate respondent IDs and invalid target values
# - Filled missing numerical values using the median
# - Filled missing categorical values using the mode
# - One-hot encoded categorical variables
# - Didn't one-hot encodeQ6 occupation codes for future O*NET matching
