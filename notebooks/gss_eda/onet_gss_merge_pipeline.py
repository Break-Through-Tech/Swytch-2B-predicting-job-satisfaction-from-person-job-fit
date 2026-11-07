import pandas as pd

#Load Data
df_gss = pd.read_csv('cleaned_gss_data.csv', dtype={'OCC10': str})
df_fsu = pd.read_csv('onet_data/FSU_data.csv', dtype={'census_num_2010': str})

# FRAGMENTATION 
# using the 6 core O*NET Work Values for predicting Job Satisfaction
onet_features = [
    'wv1val_mean', 
    'wv2val_mean', 
    'wv3val_mean', 
    'wv4val_mean', 
    'wv5val_mean', 
    'wv6val_mean'
]

# Subset the massive FSU dataframe FIRST to save memory
fsu_subset = df_fsu[['census_num_2010'] + onet_features].copy()

# dummy variables
# Assign a 1 if this job requires a moderate-to-high level of this Work Value, and a 0 if it is not important.
for col in onet_features:
    fsu_subset[f'{col}_High'] = (fsu_subset[col] >= 3.0).astype(int)

# Keep only the merge key and the new High/Low binary columns
fsu_clean = fsu_subset[['census_num_2010'] + [f'{col}_High' for col in onet_features]]


#  MERGE AND TARGET 
df_merged = pd.merge(df_gss, fsu_clean, left_on='OCC10', right_on='census_num_2010', how='left')
df_merged = df_merged.drop(columns=['census_num_2010'])

# Convert SATJOB to binary (Adjust the string to match whatever is in your CSV)
# If your CSV already has SATJOB as 1.0 and 2.0, use the exact value that means 'Very Satisfied'
if 'SATJOB' in df_merged.columns:
    df_merged['SATJOB_BINARY'] = (df_merged['SATJOB'] == 'Very Satisfied').astype(int)
    df_merged = df_merged.drop(columns=['SATJOB', 'OCC10'])
elif 'OCC10' in df_merged.columns:
    df_merged = df_merged.drop(columns=['OCC10'])


# FORCE ONE-HOT ENCODING FOR NUMERIC CATEGORIES 
# List every GSS column that is a category, not a continuous math number
categorical_columns = [
    'RACE', 'MARITAL', 'RINCOM16', 'INCOME16', 'CLASS', 
    'WRKSTAT', 'UNION', 'HEALTH', 'HAPPY', 'LIFE', 'REGION', 'PARTYID'
]

# convert categorical columns to strings 
for col in categorical_columns:
    if col in df_merged.columns:
        # We replace ".0" to make the final column names cleaner (e.g. RACE_1 instead of RACE_1.0)
        df_merged[col] = df_merged[col].astype(str).str.replace('.0', '', regex=False)

df_ml_ready = pd.get_dummies(df_merged, drop_first=True)

# Export your final dataset
df_ml_ready.to_csv('final_ml_dataset.csv', index=False)
print("Pipeline complete! Data exported to final_ml_dataset.csv")

'''
Checkpoint 
- bridged gap between GSS survey respondents and their occupational skills by merging onet and gss
datasets using Census 2010 codes. 
- transformed continuous onet features into binary dummy variables 
- performed one-hot-encoding on categorical survey data so model will not treat variables as 
math expressions. 
- produced clean final dataset ready for training and forming baseline model with O*Net and GSS
'''