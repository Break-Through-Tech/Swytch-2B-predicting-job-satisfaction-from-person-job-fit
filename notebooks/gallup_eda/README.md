# Gallup occupation linkage

`occupation_linkage.py` performs the first linkage stage for the Gallup
American Job Quality Study. It uses exact normalized source codes only. It
does not match write-in text or drop respondents whose occupation is missing
or absent from the crosswalk.

The crosswalk column names are required on the command line because the
workbook schema must be verified before running the pipeline:

```bash
python notebooks/gallup_eda/occupation_linkage.py \
  --respondents data/interim/gallup_steps1to3.parquet \
  --crosswalk /path/to/Occupation_Handbook_to_ONET-SOC.xlsx \
  --output data/interim/gallup_occupation_linked.parquet \
  --audit-output data/interim/gallup_linkage_audit.csv \
  --respondent-code-col Q6_CODED \
  --source-code-col SOURCE_CODE \
  --onet-code-col ONET_SOC_CODE \
  --source-code-width 4 \
  --attribute work_value_1 \
  --attribute work_value_2
```

The linked Parquet file retains every respondent and adds the normalized code,
O*NET code candidates, aggregated numeric attributes, `match_status`,
`match_method`, `candidate_count`, and `match_confidence`. A `.csv` audit path
writes separate overall and source-code-level coverage files. An `.xlsx` audit
path writes both tables as workbook sheets when an Excel writer is installed. A
metadata JSON file records input and crosswalk SHA-256 hashes, the selected
sheet, code columns, and the exact-code-only scope.

Run the focused tests without installing extra tooling:

```bash
python -m unittest discover -s notebooks/gallup_eda -p 'test_*.py' -v
```
