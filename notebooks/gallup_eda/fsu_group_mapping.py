"""Provisional broad Gallup group assignments for FSU Census2010 occupations.
These are code-range approximations requiring review, NOT exact respondent matches.
"""
from pathlib import Path
import re, zipfile
import numpy as np
import pandas as pd
from lxml import etree as ET

# 2010 Census occupation code blocks, aligned to Gallup Q6 labels; exceptions handled below.
BLOCKS={1:(10,430),2:(500,950),3:(1000,1240),4:(1300,1560),5:(1600,1965),6:(2000,2060),7:(2100,2160),8:(2200,2550),9:(2600,2960),12:(3600,3655),13:(3700,3955),14:(4000,4160),15:(4200,4250),16:(4300,4650),17:(4700,4965),18:(5000,5940),19:(6005,6130),20:(6200,6940),21:(7000,7630),22:(7700,8965),23:(9000,9750),25:(9800,9830)}
# Gallup distinguishes practitioners and technicians, while Census2010 puts both
# in 3000-3540. Use title-level conservative categorization, not false precision.
TECH_RE=re.compile(r'\b(technician|technologist|assistant|aide|paramedic|emergency medical|licensed practical|vocational nurse|phlebotomist|optician|medical records|diagnostic medical sonographer)\b',re.I)

def read_fsu_selected(path):
    """Stream XLSX XML; extract only occupational IDs/titles and value means."""
    with zipfile.ZipFile(path) as z:
        shared=[]
        with z.open('xl/sharedStrings.xml') as fp:
            for _,el in ET.iterparse(fp,events=('end',),tag='{*}si'):
                shared.append(''.join(el.itertext()));el.clear()
        rows=[]; headers=None; selected=None
        with z.open('xl/worksheets/sheet1.xml') as fp:
            for _,row in ET.iterparse(fp,events=('end',),tag='{*}row'):
                rn=row.get('r')
                if rn is None:continue
                cells={}
                for c in row:
                    if not c.tag.endswith('}c'):continue
                    v=c.find('{*}v');is_=c.find('{*}is')
                    value=v.text if v is not None else ''.join(is_.itertext()) if is_ is not None else None
                    if value is not None and c.get('t')=='s':value=shared[int(value)]
                    if value is not None:
                        col=re.match(r'[A-Z]+',c.get('r')).group()
                        ix=0
                        for ch in col:ix=ix*26+ord(ch)-64
                        cells[ix-1]=value
                if headers is None:
                    headers=cells
                    selected={i:n for i,n in headers.items() if i in (0,1) or n.endswith('val_mean')}
                else:
                    rows.append({name:cells.get(i) for i,name in selected.items()})
                row.clear()
    df=pd.DataFrame(rows)
    df['census_num_2010']=pd.to_numeric(df['census_num_2010'],errors='coerce')
    for c in df.columns[2:]:df[c]=pd.to_numeric(df[c],errors='coerce')
    return df

def group_for(code,title):
    if pd.isna(code):return None
    code=int(code)
    if 3000<=code<=3540:
        if code==3540: return None  # ambiguous mixed practitioner/technical Census category
        return 11 if code>=3300 else 10  # Census 2010 3300+ healthcare technical codes
    for group,(lo,hi) in BLOCKS.items():
        if lo<=code<=hi:return group
    return None

