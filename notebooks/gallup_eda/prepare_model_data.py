"""Prepare Gallup + FSU/O*NET group-level features using teammate linkage functions.
Run from any directory: python prepare_model_data.py --gallup PATH --fsu PATH
No models are trained; output is a pre-imputation, pre-encoding modeling table.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cleaning_steps import reconcile_missing_values, step2_clean_target, step3_clean_occupation
from occupation_linkage import link_respondents, linkage_diagnostics
from fsu_group_mapping import read_fsu_selected, group_for

SURVEY = ["Q8","Q9","Q14","Q16","Q17","Q20","Q23","Q24","Q27","Q28","Q31A","Q31B","Q31C","Q32A","Q32B","Q32C","Q40","Q51","Q52","Q57"]

def build(gallup_path: Path, fsu_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(gallup_path, encoding="cp1252", low_memory=False)
    clean = step3_clean_occupation(step2_clean_target(reconcile_missing_values(raw)))
    fsu = read_fsu_selected(fsu_path)
    fsu["gallup_group"] = pd.array([group_for(code, title) for code,title in zip(fsu["census_num_2010"],fsu["census_occ_title"])], dtype="Int64")
    fsu["census2010_code"] = fsu["census_num_2010"].astype("Int64").astype("string")
    fsu[["census_num_2010","census_occ_title","gallup_group"]].to_csv(output_dir/"census_to_gallup_mapping_audit.csv",index=False)
    # Reuse teammate's prepare_crosswalk/link_respondents logic to aggregate
    # one-to-many occupational features before merging, preserving respondent rows.
    fsu["gallup_group_code"] = fsu["gallup_group"].astype("string")
    clean["gallup_group_code"] = clean["gallup_occupation_group"].astype("string")
    numeric = [c for c in fsu if c.endswith("val_mean")]
    # Drop features without sufficient coverage across broad occupational groups.
    group_means = fsu.dropna(subset=["gallup_group"]).groupby("gallup_group")[numeric].mean()
    numeric = [c for c in numeric if group_means[c].notna().sum()>=15]
    fsu = fsu.rename(columns={c:"onet_"+c for c in numeric})
    attributes = ["onet_"+c for c in numeric]
    linked = link_respondents(clean, fsu, respondent_code_col="gallup_group_code", source_code_col="gallup_group_code", onet_code_col="census2010_code", numeric_attributes=attributes)
    linked = linked.rename(columns={"onet_code":"census2010_candidates"})
    linked["match_method"] = np.where(linked["match_status"].eq("matched"),"provisional_group_aggregate","none")
    linked["match_confidence"] = np.nan  # Cannot claim exact individual match confidence.
    linked["match_status"] = linked["match_status"].replace({"matched":"group_features_attached"})
    overall, by_group = linkage_diagnostics(linked.assign(match_status=linked["match_status"].replace({"group_features_attached":"matched"})),respondent_code_col="gallup_group_code")
    overall.to_csv(output_dir/"linkage_overall.csv",index=False)
    by_group.to_csv(output_dir/"linkage_by_group.csv",index=False)
    audit_cols = ["ENTITY_ID","Q6","Q6_CODED","gallup_occupation_group","occupation_source","match_status","match_method","candidate_count","census2010_candidates"]
    linked[[c for c in audit_cols if c in linked]].to_csv(output_dir/"respondent_linkage_audit.csv",index=False)
    survey=[c for c in SURVEY if c in linked]
    model_cols=["ENTITY_ID","Q4","satisfied","gallup_occupation_group","occupation_source","match_status"]+survey+attributes
    model=linked[model_cols].copy()
    for c in survey: model[c]=pd.to_numeric(model[c],errors="coerce").replace(-98,np.nan)
    model.to_csv(output_dir/"gallup_fsu_pre_model.csv",index=False)
    report={"input_rows":len(raw),"valid_target_rows":len(clean),"output_rows":len(model),"group_features_attached":int(linked["match_status"].eq("group_features_attached").sum()),"unmatched_rows":int(linked["match_status"].eq("unmatched").sum()),"feature_columns":len(attributes),"limitations":["Broad group-level FSU attributes; no individual occupation match", "Census-to-Gallup range mapping is provisional and must be reviewed", "Equal-weight aggregation across Census occupations", "FSU abbreviated features need a codebook", "No modeling, imputation or encoding performed"]}
    assert len(model)==len(clean)==len(linked)
    assert model.ENTITY_ID.is_unique
    (output_dir/"validation_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return model

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--gallup",type=Path,required=True)
    parser.add_argument("--fsu",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"output")
    args=parser.parse_args()
    build(args.gallup,args.fsu,args.output_dir)
