from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'creditcard.csv'
OUT = ROOT / 'models'
OUT.mkdir(exist_ok=True)
if not DATA.exists():
    raise FileNotFoundError(f'Place creditcard.csv at {DATA}')
df = pd.read_csv(DATA)
features = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
assert all(c in df.columns for c in features + ['Class'])
X,y = df[features],df['Class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print('Training Logistic Regression...', flush=True)
lr=LogisticRegression(max_iter=1000,random_state=42)
lr.fit(X_train_scaled,y_train)
print('Training Random Forest...',flush=True)
rf=RandomForestClassifier(n_estimators=50,max_depth=10,random_state=42,n_jobs=-1)
rf.fit(X_train_scaled,y_train)
result={}
for key,model in [('Logistic Regression',lr),('Random Forest',rf)]:
    pred=model.predict(X_test_scaled)
    prob=model.predict_proba(X_test_scaled)[:,1]
    result[key]={
        'accuracy':float(accuracy_score(y_test,pred)),
        'precision':float(precision_score(y_test,pred,zero_division=0)),
        'recall':float(recall_score(y_test,pred,zero_division=0)),
        'f1':float(f1_score(y_test,pred,zero_division=0)),
        'roc_auc':float(roc_auc_score(y_test,prob)),
        'confusion_matrix':confusion_matrix(y_test,pred).tolist()
    }
    print(key, result[key],flush=True)
joblib.dump({'scaler':scaler,'models':{'Random Forest':rf,'Logistic Regression':lr},'features':features},OUT/'fraud_bundle.joblib',compress=3)
# Keep genuine and fraudulent examples exclusively from the held-out test set.
examples=pd.concat([X_test.loc[y_test.eq(0)].head(8),X_test.loc[y_test.eq(1)].head(8)])
examples['Class']=y_test.loc[examples.index]
examples.to_csv(ROOT/'data'/'demo_transactions.csv',index=False)
metrics={'n_rows':int(len(df)),'fraud_count':int(y.sum()),'genuine_count':int((y==0).sum()),'test_count':int(len(y_test)),'fraud_test_count':int(y_test.sum()),'models':result}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
print('Saved model bundle, metrics, held-out demo transactions.',flush=True)
