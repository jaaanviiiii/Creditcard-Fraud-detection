from pathlib import Path
import json
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='FraudShield | Credit Card Fraud Detection',page_icon='🛡️',layout='wide')
st.markdown('''<style>
/* Presentation mode: hide Streamlit's Deploy, top-right menu and toolbar. */
.stAppDeployButton, [data-testid="stAppDeployButton"],
[data-testid="stToolbar"], [data-testid="stMainMenu"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, header[data-testid="stHeader"] {display:none !important; visibility:hidden !important;}
footer {display:none !important;}
.stApp {background: #0a1020;color:#f2f5ff;}
.block-container {padding-top: 1.4rem;max-width:1200px;}
.hero {padding:28px 32px;border:1px solid #263859;border-radius:22px;background:linear-gradient(115deg,#142446,#101629);margin-bottom:20px;}
.eyebrow {font-size:13px;font-weight:700;color:#81d1ff;letter-spacing:2px;}
.hero h1 {font-size:38px;color:#f8fbff;margin:8px 0;}
.hero p {color:#b3c7e4;margin:0;}
.result {padding:22px;border-radius:16px;border:1px solid #375078;background:#15233b;}
.fraud {border-color:#ef7474;background:#3a1c2b;}
.safe {border-color:#5dd7b8;background:#12332e;}
.small {color:#acbad2;font-size:14px;}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load(ROOT/'models'/'fraud_bundle.joblib')

@st.cache_data
def load_examples():
    return pd.read_csv(ROOT/'data'/'demo_transactions.csv')

@st.cache_data
def load_metrics():
    return json.loads((ROOT/'models'/'metrics.json').read_text())

bundle=load_assets()
examples=load_examples()
metrics=load_metrics()
features=bundle['features']
models=bundle['models']
scaler=bundle['scaler']

st.markdown('''<div class="hero"><div class="eyebrow">MACHINE LEARNING • LIVE DEMONSTRATION</div><h1>🛡️ FraudShield</h1><p>Credit Card Fraud Detection • Logistic Regression vs Random Forest • Real dataset and trained classifiers</p></div>''',unsafe_allow_html=True)
nav=st.tabs(['🔍 Live Fraud Detector','📊 Model Performance','📁 Bulk CSV Prediction','📘 Project Explanation'])

with nav[0]:
    c1,c2=st.columns([1.12,0.88],gap='large')
    with c1:
        st.subheader('Test a transaction')
        chosen=st.radio('Transaction source',['Sample genuine transaction','Sample fraudulent transaction','Edit all 30 feature values'],horizontal=False)
        model_name=st.selectbox('Choose ML model',list(models),index=0)
        if chosen.startswith('Sample'):
            flag=int('fraudulent' in chosen)
            subset=examples[examples['Class']==flag].reset_index(drop=True)
            idx=st.selectbox('Choose a held-out test example',range(len(subset)),format_func=lambda i:f'Example {i+1} — Amount: {subset.iloc[i]["Amount"]:,.2f}')
            row=subset.loc[idx,features].copy()
            st.caption('These examples were NOT used to train the models. The known label is used only for demonstration.')
            with st.expander('View transaction feature values'):
                st.dataframe(row.rename('Value'),use_container_width=True)
        else:
            st.caption('V1–V28 are anonymized numerical features. Editing them creates a hypothetical feature vector, not a real card transaction.')
            default=examples.loc[0,features]
            with st.form('manual'):
                a,b=st.columns(2)
                vals={}
                for i,f in enumerate(features):
                    with (a if i%2==0 else b):
                        vals[f]=st.number_input(f,value=float(default[f]),format='%.6f')
                submitted=st.form_submit_button('Apply feature values',use_container_width=True)
            row=pd.Series(vals)
        go=st.button('🔎 Detect Fraud',type='primary',use_container_width=True)
    with c2:
        st.subheader('Detection result')
        if go:
            x=pd.DataFrame([[float(row[f]) for f in features]],columns=features)
            scaled=scaler.transform(x)
            pred=int(models[model_name].predict(scaled)[0])
            p=float(models[model_name].predict_proba(scaled)[0,1])
            st.markdown(f'''<div class="result {'fraud' if pred else 'safe'}"><div class="small">MODEL PREDICTION</div><h2>{'🚨 FRAUD SUSPECTED' if pred else '✅ GENUINE TRANSACTION'}</h2><h3>{p*100:.2f}% estimated fraud probability</h3><p>{'This transaction is flagged for review.' if pred else 'The classifier considers this transaction low risk.'}</p></div>''',unsafe_allow_html=True)
            st.progress(min(max(p,0.),1.))
            st.write('**Selected algorithm:**',model_name)
            if chosen.startswith('Sample'):
                actual=int('fraudulent' in chosen)
                st.write('**Known test label:**','Fraud' if actual else 'Genuine')
                st.write('**Matches known label:**','Yes ✅' if pred==actual else 'No ⚠️')
            comparison=[]
            for name,model in models.items():
                comparison.append({'Model':name,'Fraud probability':f'{model.predict_proba(scaled)[0,1]*100:.2f}%','Prediction':'Fraud' if model.predict(scaled)[0] else 'Genuine'})
            st.write('**Both models on this transaction**')
            st.dataframe(pd.DataFrame(comparison),hide_index=True,use_container_width=True)
        else:
            st.info('Choose a sample on the left, then click **Detect Fraud** to display the prediction.')
        st.caption('Educational demonstration only. A model prediction is not proof of fraud and should not be used alone for financial decisions.')

with nav[1]:
    st.subheader('Evaluation on unseen test transactions')
    a,b,c,d=st.columns(4)
    a.metric('Dataset transactions',f'{metrics["n_rows"]:,}')
    b.metric('Fraud transactions',f'{metrics["fraud_count"]:,}')
    c.metric('Test transactions',f'{metrics["test_count"]:,}')
    d.metric('Test fraud cases',f'{metrics["fraud_test_count"]:,}')
    rows=[]
    for name,m in metrics['models'].items():
        rows.append({'Algorithm':name,'Accuracy':round(m['accuracy']*100,3),'Precision':round(m['precision']*100,2),'Recall':round(m['recall']*100,2),'F1 score':round(m['f1']*100,2),'ROC AUC':round(m['roc_auc'],4)})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.write('**Compare fraud detection recall and precision (%)**')
    st.bar_chart(pd.DataFrame(rows).set_index('Algorithm')[['Precision','Recall','F1 score']])
    st.write('**Confusion matrices**')
    l,r=st.columns(2)
    for col,(name,m) in zip([l,r],metrics['models'].items()):
        with col:
            st.markdown(f'**{name}**')
            cm=np.array(m['confusion_matrix'])
            st.dataframe(pd.DataFrame(cm,index=['Actual genuine','Actual fraud'],columns=['Predicted genuine','Predicted fraud']),use_container_width=True)
            st.caption(f'Fraud caught: {cm[1,1]}/{cm[1].sum()} • Missed fraud: {cm[1,0]} • False alarms: {cm[0,1]}')
    st.info('Accuracy can be misleading because only about 0.17% of all transactions are fraud. Recall and precision matter when explaining model performance.')

with nav[2]:
    st.subheader('Predict several transactions at once')
    st.write('Upload a CSV containing all 30 features: `Time`, `V1`–`V28`, and `Amount`. The optional `Class` column is ignored for prediction.')
    st.download_button('⬇️ Download sample input CSV',examples[features].to_csv(index=False),file_name='example_input.csv',mime='text/csv')
    uploaded=st.file_uploader('Choose transaction CSV',type=['csv'])
    bulk_model=st.selectbox('Model for CSV predictions',list(models),key='bulk')
    if uploaded:
        try:
            df=pd.read_csv(uploaded)
            absent=[f for f in features if f not in df.columns]
            if absent:
                st.error('Missing columns: '+', '.join(absent))
            elif len(df)==0:
                st.error('The file has no transactions.')
            elif len(df)>20000:
                st.error('For a smooth class demo, upload at most 20,000 rows at a time.')
            else:
                X=df[features].apply(pd.to_numeric,errors='coerce')
                if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
                    st.error('All feature values must be valid, finite numbers.')
                else:
                    transformed=scaler.transform(X)
                    probabilities=models[bulk_model].predict_proba(transformed)[:,1]
                    preds=models[bulk_model].predict(transformed)
                    result=df.copy()
                    result['Fraud_Probability']=probabilities.round(5)
                    result['Prediction']=np.where(preds==1,'Fraud','Genuine')
                    st.metric('Transactions flagged as fraud',int(preds.sum()))
                    st.dataframe(result.head(100),use_container_width=True)
                    st.download_button('⬇️ Download prediction results',result.to_csv(index=False),file_name='fraud_predictions.csv',mime='text/csv')
        except Exception as e:
            st.error(f'Unable to process CSV: {e}')

with nav[3]:
    st.subheader('How this project works')
    st.markdown('''**1. Problem:** Detect suspicious card transactions using historical labeled data.

**2. Dataset:** 284,807 anonymized card transactions with 30 numeric inputs and a target label `Class` (0 = genuine, 1 = fraud).

**3. Preprocessing:** An 80/20 stratified train/test split preserves class proportions. `StandardScaler` is fitted on training data only and then applied to testing data.

**4. Machine learning:** Logistic Regression finds a linear decision boundary; Random Forest combines multiple decision trees.

**5. Prediction:** The app sends a feature vector through the same saved scaler and chosen trained model, then displays its prediction and class probability.

**6. Evaluation:** Confusion matrix, accuracy, precision, recall, F1 and ROC AUC are calculated on unseen test data.

**Why not enter a card number?** This public dataset intentionally does not contain actual card details. The 28 `V` features are anonymized transformations. You cannot accurately derive them from a card number, customer name or transaction amount alone.

**For your viva:** Say: “I trained two supervised machine learning models on labeled transactions. I tested them on unseen data and built a live interface where a test transaction is classified as genuine or fraudulent.”''')
    st.caption('Implementation follows the supplied Colab notebook: stratified split, StandardScaler, Logistic Regression, and 50-tree depth-10 Random Forest.')
