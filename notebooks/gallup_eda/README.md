# Gallup + FSU/O*NET preparation (updated teammate pipeline)

**Existing files retained:** `cleaning_steps.py`, `occupation_linkage.py`, `test_occupation_linkage.py`. The cleaning script is corrected to use the numeric Gallup satisfaction column and derive a Gallup occupation group. The original linkage functions are reused by `prepare_model_data.py`, not replaced.

## Run

```bash
python -m unittest discover -s . -p 'test_*.py' -v
python prepare_model_data.py --gallup /path/to/JFF_formatted_public_release_values.csv --fsu /path/to/Census2010-O_NET26.1_Data.xlsx
```

Outputs in `output/`: `gallup_fsu_pre_model.csv`, `respondent_linkage_audit.csv`, `census_to_gallup_mapping_audit.csv`, `linkage_overall.csv`, `linkage_by_group.csv`, `validation_report.json`.

## Critical limitations

- FSU contains detailed **2010 Census occupation codes**, while Gallup Q6 contains broad categories. We use a **provisional code-range grouping** (including a healthcare subgroup approximation), not exact Gallup-to-Census matching. Inspect `census_to_gallup_mapping_audit.csv` and validate all groups before presenting as a final scientific crosswalk.
- O*NET features are **unweighted group averages**, not respondent-specific occupation features. No false `match_confidence=1` is reported.
- Q4/satisfied is the target; never include Q4, ENTITY_ID, or audit columns as predictors.
- **No model is trained here.** Split train/test before fitting any imputer, scaler, encoder, or feature selector. Review potential target leakage in survey predictors.
- Original `occupation_linkage.py` CLI remains available for genuine exact-code crosswalks, but should **not** be used directly with Gallup Q6 and the OOH crosswalk.
