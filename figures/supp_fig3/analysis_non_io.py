import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_survival':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_survival')]
import sys
sys.path.append(str(cwd))
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd
import pickle
from model import utils
from matplotlib import pyplot as plt

t = utils.LogTransform(bias=4, min_x=0)
cph = CoxPHFitter()

df = pd.read_csv(cwd / 'figures' / 'supp_fig3'/ 'data' / 'TMB_public.csv', sep=',', low_memory=False)
df = df.loc[df['Type of ICI treatment'] == 'Never received ICI']
df['OS'] = (df['Vital status'] == 'DECEASED').astype(np.int32)

for cancer in ['NSCLC', 'Colorectal', 'Pancreatic', 'Endometrial']:
    print(cancer)
    cancer_df = df.loc[df['Cancer type'] == cancer]
    tmb = cancer_df['TMB (mutations/Mb)'].values
    cancer_df = cancer_df.loc[tmb < np.percentile(tmb, 99)]
    tmb = t.trf(cancer_df['TMB (mutations/Mb)'].values)
    times = cancer_df['Overall Survival from diagnosis (Months)'].values
    events = cancer_df['OS'].values
    print(len(cancer_df))

    test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig3' / (cancer + '_non_io.pkl'), 'rb'))

    cox_losses = []
    for idx_test in test_idx:
        mask = np.ones(len(cancer_df), dtype=bool)
        mask[idx_test] = False
        cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
        cox_losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': tmb[idx_test]})))

    print('cox')
    print(round(np.mean(cox_losses), 3))
    print(round(cph.fit(pd.DataFrame({'T': times, 'E': events, 'x': tmb}), 'T', 'E', formula='x').concordance_index_, 3))

    for model in ['FCN']:
        print(model)
        concordance = concordance_index(times[np.concatenate(test_idx)], np.concatenate(results[model][0]), events[np.concatenate(test_idx)])
        losses = []
        for idx_test, risks in zip(test_idx, results[model][1]):
            mask = np.ones(len(risks), dtype=bool)
            mask[idx_test] = False
            cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
            losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
        print(round(np.mean(losses), 3))
        print(round(concordance, 3))

