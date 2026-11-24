"""ASK-T2D exploratory NHANES cross-sectional screening benchmark; not clinical advice."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

ROOT="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
PERIODS={
 "development":("2015",["DEMO_I","BMX_I","GHB_I","DIQ_I"]),
 "temporal_test":("2017",["P_DEMO","P_BMX","P_GHB","P_DIQ"])
}
BASE=["RIDAGEYR","RIAGENDR"]; EXTRA=["BMXBMI","BMXWAIST"]

def read_xpt(year,name):
    p=Path("data")/(name+".xpt");p.parent.mkdir(exist_ok=True)
    if not p.exists():
        import requests
        url=ROOT.format(year=year,name=name)
        response=requests.get(url,timeout=90)
        response.raise_for_status()
        if len(response.content)<1000: raise RuntimeError("Unexpectedly small CDC file: "+url)
        p.write_bytes(response.content)
    return pd.read_sas(p,format="xport")

def load(period):
    year,names=PERIODS[period]
    frames=[read_xpt(year,n) for n in names]
    df=frames[0]
    for frame in frames[1:]:df=df.merge(frame,on="SEQN",how="left",validate="one_to_one")
    cols=BASE+EXTRA+["LBXGH","DIQ010"]
    missing=set(cols)-set(df)
    if missing:raise ValueError("NHANES missing columns: "+repr(missing))
    df=df[(df["RIDAGEYR"]>=20)&df["LBXGH"].notna()&(df["DIQ010"]==2)].copy()
    df["target"]=(df["LBXGH"]>=6.5).astype(int)
    if df["target"].nunique()!=2:raise ValueError("Cohort has only one class")
    return df

def probabilities(a,b,columns):
    pipeline=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
    pipeline.fit(a[columns],a["target"])
    return pipeline.predict_proba(b[columns])[:,1]

def evaluate(y,p):
    return {"n":int(len(y)),"positive_count":int(sum(y)),"auroc":round(float(roc_auc_score(y,p)),4),
    "average_precision":round(float(average_precision_score(y,p)),4),
    "brier":round(float(brier_score_loss(y,p)),4),
    "log_loss":round(float(log_loss(y,p,labels=[0,1])),4)}

def main():
    dev=load("development");test=load("temporal_test")
    tr,va=train_test_split(dev,test_size=0.25,stratify=dev["target"],random_state=2026)
    options={}
    for feature in EXTRA:
        options[feature]={"validation":evaluate(va["target"],probabilities(tr,va,BASE+[feature])),
        "temporal_test":evaluate(test["target"],probabilities(tr,test,BASE+[feature]))}
    chosen=min(EXTRA,key=lambda feature:options[feature]["validation"]["brier"])
    result={"status":"Executed NHANES methodological benchmark, NOT clinically validated",
      "development_cohort_n":len(dev),"temporal_test_cohort_n":len(test),
      "method":"HbA1c reference >=6.5%; age >=20; DIQ010=2; HbA1c excluded from predictors",
      "baseline":evaluate(test["target"],probabilities(tr,test,BASE)),
      "single_measurement_options":options,"chosen_on_development_validation_only":chosen,
      "all_measurements":evaluate(test["target"],probabilities(tr,test,BASE+EXTRA)),
      "limitations":["Not an adaptive patient-specific testing strategy","No measurement cost or clinical net benefit model",
      "No survey-weighted population inference","No clinical diagnosis or validation","No independent hospital cohort"]}
    path=Path("results/nhanes_results.json");path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
