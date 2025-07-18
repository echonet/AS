import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, f1_score

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -100, 100)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


models = ['PLAX', 'PSAX', 'Apical', 'PLAX_D', 'PSAX_D', 'Apical_D']
true_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
logit_cols = ["no_preds", "mild_preds", "mild~moderate_preds", "moderate_preds", "moderate~severe_preds", "severe_preds"]
prob_cols = ["no_p", "mild_p", "mild_mod_p", "mod_p", "mod_sev_p", "sev_p"]
phenotype = 'AS_severity'
# peakav_df = pd.read_csv('./peakav.csv', dtype={'study_uid': str, 'peakav': float}) # Uncomment this above line if you have a peakav.csv file

def main():
    dfs = [pd.read_csv(f"./predictions/{m}.csv", dtype={'study_uid': str}, low_memory=False).assign(model=m).drop_duplicates() for m in models]
    df = pd.concat(dfs, ignore_index=True)

    df[prob_cols] = softmax(df[logit_cols].to_numpy())
    agg = df.groupby(["study_uid", "model"])[prob_cols].mean()
    y6 = df.groupby("study_uid")[phenotype].first().to_numpy()
    y4 = np.vectorize(true_map.get)(y6)
    wide = agg.unstack("model")
    wide.columns = [f"{mdl}_{col}" for col, mdl in wide.columns]

    wide.loc[:,'peakav'] = np.nan # change this to the actual peakav values if available
    # wide = wide.merge(peakav_df.set_index('study_uid'), left_index=True, right_index=True, how='left') 

    with open('./weights/av_stenosis_peakav.pkl', 'rb') as f:
        ens = pickle.load(f)
    prob = ens.predict_proba(wide.to_numpy())
    print('N:', prob.shape[0])
    print('accuracy:', np.mean(np.argmax(prob, axis=1) == y4))
    print('f1:', f1_score(y4, np.argmax(prob, axis=1), average='weighted', zero_division=0))
    print('auc_macro:', roc_auc_score(y4, prob, multi_class="ovr", average="macro"))

if __name__ == "__main__":
    main()