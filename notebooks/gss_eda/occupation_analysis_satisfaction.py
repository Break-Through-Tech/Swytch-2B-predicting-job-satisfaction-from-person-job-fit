import pandas as pd
from scipy.stats import chi2_contingency

# 1. Load cleaned GSS data (which still has OCC10)
df = pd.read_csv('cleaned_gss_data.csv')

# Create binary target (adjust if SATJOB is already numeric/binary in your file)
if 'SATJOB_BINARY' not in df.columns:
    df['SATJOB_BINARY'] = (df['SATJOB'] == 'Very Satisfied').astype(int)

df['OCC10_num'] = pd.to_numeric(df['OCC10'], errors='coerce')
df = df.dropna(subset=['OCC10_num', 'SATJOB_BINARY'])

# 2. Map OCC10 into Standard Census 2010 Major Occupation Groups
def map_occ_group(code):
    if 10 <= code <= 3540:
        return 'Management, Business, Science & Arts'
    elif 3600 <= code <= 4650:
        return 'Service Occupations'
    elif 4700 <= code <= 5940:
        return 'Sales & Office'
    elif 6000 <= code <= 7630:
        return 'Natural Resources, Construction & Maintenance'
    elif 7700 <= code <= 9750:
        return 'Production, Transportation & Material Moving'
    else:
        return 'Military / Other'

df['OCC_GROUP'] = df['OCC10_num'].apply(map_occ_group)

# 3. CHI-SQUARE TEST (Is the difference statistically significant?)
contingency_table = pd.crosstab(df['OCC_GROUP'], df['SATJOB_BINARY'])
chi2, p_val, dof, ex = chi2_contingency(contingency_table)

print(f"--- Chi-Square Test (Occupation Group vs. Job Satisfaction) ---")
print(f"Chi2 Statistic: {chi2:.4f} | p-value: {p_val:.4e}\n")

# 4. SATISFACTION RATE BY BROAD OCCUPATION GROUP
group_summary = df.groupby('OCC_GROUP')['SATJOB_BINARY'].agg(
    Respondents='count',
    Pct_Very_Satisfied=lambda x: round(x.mean() * 100, 2)
).sort_values(by='Pct_Very_Satisfied', ascending=False)

print("--- Job Satisfaction by Major Occupation Group ---")
print(group_summary)

# 5. TOP 5 HIGHEST & LOWEST SPECIFIC OCC10 CODES (Filtered to n >= 30)
occ_summary = df.groupby('OCC10_num')['SATJOB_BINARY'].agg(
    Respondents='count',
    Pct_Very_Satisfied=lambda x: round(x.mean() * 100, 2)
)
occ_filtered = occ_summary[occ_summary['Respondents'] >= 30].sort_values(
    by='Pct_Very_Satisfied', ascending=False
)

print("\n--- Top 5 Most Satisfied OCC10 Codes (n >= 30) ---")
print(occ_filtered.head(5))

print("\n--- Bottom 5 Least Satisfied OCC10 Codes (n >= 30) ---")
print(occ_filtered.tail(5))