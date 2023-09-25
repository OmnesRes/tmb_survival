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

labels_to_use = ['BLCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC']

data = pickle.load(open(cwd / 'files' / 'data.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
tmb_dict = {i[:12]: data[i][0] / (data[i][1] / 1e6) for i in data}

samples['tmb'] = samples.bcr_patient_barcode.apply(lambda x: tmb_dict.get(x, np.nan))
samples.dropna(axis=0, subset=['OS', 'OS.time', 'tmb'], inplace=True)

t = utils.LogTransform(bias=4, min_x=0)

cph = CoxPHFitter()
for cancer in labels_to_use:
    print(cancer)
    df = samples.loc[samples['type'] == cancer]
    tmb = df['tmb'].values
    df = df.loc[tmb < np.percentile(tmb, 99)]
    tmb = t.trf(df.tmb.values)[:, np.newaxis]
    times = df['OS.time'].values
    events = df['OS'].values
    for model in ['FCN', '2neuron', 'sigmoid']:
        print(model)
        test_idx, test_ranks, all_risks = pickle.load(open(cwd / 'figures' / 'fig3' / (model + '_runs.pkl'), 'rb'))[cancer]
        concordance = concordance_index(times[np.concatenate(test_idx)], np.concatenate(test_ranks), events[np.concatenate(test_idx)])
        print(round(concordance, 3))
        losses = []
        for idx_test, risks in zip(test_idx, all_risks):
            mask = np.ones(len(risks), dtype=bool)
            mask[idx_test] = False
            cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
            losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
        print(round(np.mean(losses), 3))






###cox
cox_losses = []
for idx_test in test_idx:
    mask = np.ones(len(risks), dtype=bool)
    mask[idx_test] = False
    cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
    cox_losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': tmb[idx_test]})))
print('cox')
print(round(cph.fit(pd.DataFrame({'T': times, 'E': events, 'x': tmb}), 'T', 'E', formula='x').concordance_index_, 3))
print(round(np.mean(cox_losses), 3))


###true values
print('true')
sim_risks = np.array(sim_risks)
print(round(concordance_index(times, -sim_risks, events), 3))
true_losses = []
for idx_test in test_idx:
    mask = np.ones(len(risks), dtype=bool)
    mask[idx_test] = False
    cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': sim_risks[mask]}), 'T', 'E', formula='x')
    true_losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': sim_risks[idx_test]})))
print(round(np.mean(true_losses), 3))



