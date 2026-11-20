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

print(df_fsu[onet_features].describe())
'''
 wv1val_mean  wv2val_mean  wv3val_mean  wv4val_mean  wv5val_mean  wv6val_mean
count   508.000000   508.000000   508.000000   508.000000   508.000000   508.000000
mean      3.782052     4.146714     3.333763     4.403290     4.444417     3.824384
std       1.324266     1.160985     1.226245     1.080937     0.853855     1.084978
min       1.000000     1.670000     1.000000     1.670000     2.000000     1.500000
25%       2.670000     3.310625     2.330000     3.670000     4.000000     3.000000
50%       3.670000     4.170000     3.000000     4.330000     4.443333     3.830000
75%       5.000000     5.220000     4.330000     5.000000     5.000000     4.673000
max       7.000000     6.835000     6.500000     7.000000     6.835000     6.330000
'''

# Subset the massive FSU dataframe FIRST to save memory
fsu_subset = df_fsu[['census_num_2010'] + onet_features].copy()

# dummy variables
# Assign a 1 if this job requires a moderate-to-high level of this Work Value, and a 0 if it is not important.
for col in onet_features:
    fsu_subset[f'{col}_High'] = (fsu_subset[col] >= 3.0).astype(int)

# Keep only the merge key and the new High/Low binary columns
fsu_clean = fsu_subset[['census_num_2010'] + [f'{col}_High' for col in onet_features]]
df_gss['OCC10'] = df_gss['OCC10'].astype(str).str.replace('.0', '', regex=False).str.strip()
fsu_clean['census_num_2010'] = fsu_clean['census_num_2010'].astype(str).str.replace('.0', '', regex=False).str.strip()

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