"""Synthetic-data baseline models. Evaluation uses held-out scenario seeds."""
from functools import lru_cache
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

FEATURES = ["snr_db", "rssi_dbm", "ber", "loss_pct", "latency_ms", "jitter_ms", "bandwidth_mbps", "distance_km", "doppler_hz"]


def vector(link):
    return [float(link.get(k,0)) for k in FEATURES]


def make_data(seed, count=1800):
    rng = np.random.default_rng(seed)
    snr = rng.uniform(1,28,count)
    dist = rng.uniform(500,4500,count)
    severity = np.clip((17-snr)/17 + rng.normal(0,0.12,count),0,1)
    loss = np.clip(severity*85+rng.normal(0,4,count),0,100)
    latency = dist/299.792458+4+severity*120+rng.uniform(0,3,count)
    bw = np.maximum(0.1,150-dist/55-severity*80)
    x = np.column_stack([snr,-48-dist/350,np.clip(np.exp(-snr/2),0,1),loss,latency,
                         rng.uniform(0.1,2,count)+severity*12,bw,dist,rng.uniform(-1100,1100,count)])
    y = np.where(severity > 0.82,2,np.where(severity > 0.32,1,0))
    return x,y


@lru_cache(maxsize=1)
def models():
    x_train,y_train=make_data(42)
    x_val,y_val=make_data(43,700)
    x_test,y_test=make_data(44,700)
    rf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1)
    rf.fit(x_train,y_train)
    xgb=XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.08,
                      objective="multi:softprob",num_class=3,random_state=42,n_jobs=1)
    xgb.fit(x_train,y_train)
    anomaly=IsolationForest(contamination=0.15,random_state=42,n_estimators=100)
    anomaly.fit(x_train[y_train==0])
    # Horizon labels are generated from synthetic risk processes, not orbital trajectories.
    rng=np.random.default_rng(111)
    risk=np.clip((20-x_train[:,0])/23 + x_train[:,3]/130,0,1)
    valrisk=np.clip((20-x_val[:,0])/23 + x_val[:,3]/130,0,1)
    testrisk=np.clip((20-x_test[:,0])/23 + x_test[:,3]/130,0,1)
    preds={}; metrics={}
    for horizon in [15,30,60]:
        p=np.clip(risk*(0.25+horizon/100),0.01,0.98)
        pv=np.clip(valrisk*(0.25+horizon/100),0.01,0.98)
        pt=np.clip(testrisk*(0.25+horizon/100),0.01,0.98)
        y=rng.binomial(1,p)
        yv=rng.binomial(1,pv)
        yt=rng.binomial(1,pt)
        model=XGBClassifier(n_estimators=90,max_depth=3,learning_rate=0.07,random_state=42,n_jobs=1)
        model.fit(x_train,y)
        calibrated=CalibratedClassifierCV(FrozenEstimator(model),method="sigmoid")
        calibrated.fit(x_val,yv)
        prob=calibrated.predict_proba(x_test)[:,1]
        metrics[str(horizon)]={"roc_auc":round(float(roc_auc_score(yt,prob)),4),
                               "brier":round(float(brier_score_loss(yt,prob)),4)}
        preds[horizon]=calibrated
    scores={"random_forest":{"accuracy":round(float(accuracy_score(y_test,rf.predict(x_test))),4),
                             "macro_f1":round(float(f1_score(y_test,rf.predict(x_test),average="macro")),4)},
            "xgboost":{"accuracy":round(float(accuracy_score(y_test,xgb.predict(x_test))),4),
                       "macro_f1":round(float(f1_score(y_test,xgb.predict(x_test),average="macro")),4)}}
    return rf,xgb,anomaly,preds,{"detection":scores,"prediction":metrics,
            "limitations":"Synthetic IID samples; horizon labels are probabilistic proxies, not true future orbital failure events. No temporal holdout or LSTM comparison yet."}


def infer(link):
    rf,xgb,anomaly,preds,_=models()
    x=np.array([vector(link)])
    labels=["healthy","degraded","unexpected_failure"]
    probs={str(h):round(float(m.predict_proba(x)[0,1]),4) for h,m in preds.items()}
    return {"detection":labels[int(rf.predict(x)[0])],"xgboost_detection":labels[int(xgb.predict(x)[0])],
            "anomaly":bool(anomaly.predict(x)[0]<0),"failure_probability":probs,
            "feature_importance":dict(zip(FEATURES,[round(float(v),4) for v in rf.feature_importances_]))}
