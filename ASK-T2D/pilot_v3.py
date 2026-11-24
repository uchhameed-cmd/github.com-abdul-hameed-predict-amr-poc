"""ASK-T2D: genuine, exploratory NHANES 2015-16 -> 2017-Mar2020 benchmark.
NOT diagnosis, clinical validation, treatment guidance or population prevalence estimation.
All model selection uses only development cycle; temporal test labels are never used to tune policy.
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
CYCLES = {"development":("2015",["DEMO_I","DIQ_I","GHB_I","BMX_I"]),
          "temporal_test":("2017",["P_DEMO","P_DIQ","P_GHB","P_BMX"])}
BASE=["RIDAGEYR","RIAGENDR"]
OPTIONAL=["BMXBMI","BMXWAIST"]
SEED=2026

def get_xpt(year,name):
    folder=Path("ASK-T2D/data");folder.mkdir(parents=True,exist_ok=True)
    file=folder/(name+".xpt")
    if not file.exists():
        url=ROOT.format(year=year,name=name)
        r=requests.get(url,timeout=90,headers={"User-Agent":"ASK-T2D research replication/1.0"})
        r.raise_for_status()
        if len(r.content)<1000:raise RuntimeError("Invalid or tiny CDC response: "+url)
        file.write_bytes(r.content)
    return pd.read_sas(file,format="xport")

def build_cohort(cycle):
    year,names=CYCLES[cycle]
    tables=[get_xpt(year,n) for n in names]
    x=tables[0]
    for t in tables[1:]:x=x.merge(t,on="SEQN",how="left",validate="one_to_one")
    need=BASE+OPTIONAL+["DIQ010","LBXGH"]
    if set(need)-set(x.columns):raise ValueError("Missing NHANES variables: "+str(set(need)-set(x)))
    before=len(x)
    x=x.loc[(x.RIDAGEYR>=20)&(x.DIQ010==2)&x.LBXGH.notna()].copy()
    after_filter=len(x)
    # Complete optional measurements provide a common eligibility base for equal-count tests.
    x=x.dropna(subset=OPTIONAL+BASE).copy()
    x["y"]=(x.LBXGH>=6.5).astype(int)
    if x.y.nunique()!=2:raise ValueError("Insufficient positive/negative examples")
    return x,{"all_interviewed":before,"eligible_hba1c":after_filter,"complete_body_measures":len(x),
              "positive":int(x.y.sum()),"positive_fraction":float(x.y.mean())}

def fit_model(df,features):
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,random_state=SEED))
    m.fit(df[features],df.y)
    return m

def predict(model,df,features):
    return model.predict_proba(df[features])[:,1]

def score(y,p):
    return {"auroc":float(roc_auc_score(y,p)),
     "average_precision":float(average_precision_score(y,p)),
     "brier":float(brier_score_loss(y,p)),
     "log_loss":float(log_loss(y,p,labels=[0,1])),
     "sensitivity_at_10pct_referral":float(((y==1)&(p>=np.quantile(p,.9))).sum()/max(1,(y==1).sum()))}

def fit_sampler(train,f):
    model=make_pipeline(StandardScaler(),Ridge(alpha=2.0))
    model.fit(train[BASE],train[f])
    err=train[f].to_numpy()-model.predict(train[BASE])
    return model,err

def value_score(data,aug,reg,errs,f,rng_seed):
    # Predictive variance is a proxy for expected Brier gain ONLY under calibrated Bayes assumptions.
    rng=np.random.default_rng(rng_seed)
    mean=reg.predict(data[BASE])
    draws=[]
    for _ in range(48):
        tmp=data[BASE].copy()
        tmp[f]=mean+rng.choice(errs,size=len(data),replace=True)
        draws.append(predict(aug,tmp,BASE+[f]))
    return np.asarray(draws).var(axis=0)

def policy_prediction(data,base,aug,f,mask):
    p=predict(base,data,BASE)
    p[mask]=predict(aug,data.loc[mask],BASE+[f])
    return p

def choose_k(v,k):
    chosen=np.zeros(len(v),dtype=bool)
    chosen[np.argsort(-np.asarray(v),kind="stable")[:k]]=True
    return chosen

def compare(data,base,aug,f,val,reg,errors):
    # Budget is exactly half of eligible complete-case participants, for every policy.
    k=len(data)//2
    scores=value_score(data,aug,reg,errors,f,SEED+3)
    basep=predict(base,data,BASE)
    acquisition={
       "value_proxy":choose_k(scores,k),
       "highest_risk":choose_k(basep,k),
       "highest_uncertainty":choose_k(basep*(1-basep),k),
       "random_seed_2027":choose_k(np.random.default_rng(SEED+1).random(len(data)),k)
    }
    out={}
    yy=data.y.to_numpy()
    for name,mask in acquisition.items():
        out[name]=score(yy,policy_prediction(data,base,aug,f,mask))
    # Randomisation distribution: 200 independently drawn equal-budget allocations.
    random_briers=[]
    for seed in range(200):
        rng_ref=np.random.default_rng(40000+seed)
        m=choose_k(rng_ref.random(len(data)),k)
        random_briers.append(float(brier_score_loss(yy,policy_prediction(data,base,aug,f,m))))
    out["random_200_seed_distribution"]={
      "n_seeds":200,
      "brier_mean":float(np.mean(random_briers)),
      "brier_standard_deviation":float(np.std(random_briers,ddof=1)),
      "brier_2p5pct_97p5pct":list(map(float,np.quantile(random_briers,[.025,.975]))),
      "policy_minus_mean_random_brier":float(brier_score_loss(yy,policy_prediction(data,base,aug,f,acquisition["value_proxy"]))-np.mean(random_briers)),
      "fraction_random_brier_worse_than_policy":float(np.mean(np.asarray(random_briers)>brier_score_loss(yy,policy_prediction(data,base,aug,f,acquisition["value_proxy"]))))
    }
    out["no_acquisition"]=score(yy,basep)
    out["universal_acquisition_100pct"]=score(yy,predict(aug,data,BASE+[f]))
    out["n"]=len(data);out["acquisitions_at_half_budget"]=k
    out["selected_measurement"]=f
    # Paired bootstrap CI: Brier(policy)-Brier(random), negative favors policy.
    pred_a=policy_prediction(data,base,aug,f,acquisition["value_proxy"])
    pred_r=policy_prediction(data,base,aug,f,acquisition["random_seed_2027"])
    observed=float(np.mean((yy-pred_a)**2-(yy-pred_r)**2))
    rng=np.random.default_rng(49)
    draws=[]
    for _ in range(1000):
        idx=rng.integers(0,len(yy),len(yy))
        draws.append(float(np.mean((yy[idx]-pred_a[idx])**2-(yy[idx]-pred_r[idx])**2)))
    out["paired_brier_policy_minus_random"]={"difference":observed,
        "bootstrap_95pct_percentile_interval":list(map(float,np.quantile(draws,[.025,.975])))}
    return out

def main():
    dev,dev_info=build_cohort("development")
    test,test_info=build_cohort("temporal_test")
    train,val=train_test_split(dev,test_size=.25,random_state=SEED,stratify=dev.y)
    base=fit_model(train,BASE)
    options={}
    for f in OPTIONAL:
        aug=fit_model(train,BASE+[f]);reg,err=fit_sampler(train,f)
        k=len(val)//2
        v=value_score(val,aug,reg,err,f,SEED+2)
        p=policy_prediction(val,base,aug,f,choose_k(v,k))
        options[f]={"validation_brier":float(brier_score_loss(val.y,p))}
    chosen=min(OPTIONAL,key=lambda f:options[f]["validation_brier"])
    # Preserve development-only training; once selected, refit on full development data.
    base=fit_model(dev,BASE)
    aug=fit_model(dev,BASE+[chosen])
    reg,err=fit_sampler(dev,chosen)
    result={
      "dataset":"CDC NHANES public-use XPT, nonoverlapping 2015-16 and 2017-Mar2020",
      "endpoint":"single HbA1c>=6.5%, among self-reported non-diabetic adults; not confirmed diabetes",
      "population":"age>=20, DIQ010=2, HbA1c observed, BMI and waist observed",
      "development_cohort":dev_info,"temporal_test_cohort":test_info,
      "selection_development_only":options,
      "results":compare(test,base,aug,chosen,val,reg,err),
      "limitations":["Cross-sectional endpoint; no clinical diagnosis or prospective outcome",
      "Complete-case analysis may bias results",
      "No NHANES survey design weights; no population prevalence inference",
      "Variance-based proxy NOT actual expected Brier gain",
      "200 random allocations compared; paired bootstrap remains based on one prespecified random seed",
      "United States survey; no Pakistani external validation",
      "Not clinically deployable or approved"]
    }
    output=Path("ASK-T2D/results/pilot_v3_results.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print("RESULTS_START")
    print(json.dumps(result,indent=2))
    print("RESULTS_END")

if __name__=="__main__":main()
