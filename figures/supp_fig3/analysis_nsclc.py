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

##io data
print('io data')
df = pickle.load(open(cwd / 'figures' / 'supp_fig3' / 'nsclc_sample_table.pkl', 'rb'))
df.dropna(axis=0, subset=['OS', 'OS.days', 'mean_tmb'], inplace=True)
io_drugs = 'Atezolizumab|Durvalumab|Ipilimumab|Nivolumab|Pembrolizumab'
df = df.loc[df['regimen_drugs'].str.contains(io_drugs)]
print(len(df))

tmb = df['mean_tmb'].values
df = df.loc[tmb < np.percentile(tmb, 99)]
tmb = t.trf(df.mean_tmb.values)
times = df['OS.days'].values
events = df['OS'].values

test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig3' / ('nsclc_io.pkl'), 'rb'))

cox_losses = []
for idx_test in test_idx:
    mask = np.ones(len(df), dtype=bool)
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
    

##nonio data
print('nonio data')
df = pickle.load(open(cwd / 'figures' / 'supp_fig3' / 'nsclc_sample_table.pkl', 'rb'))
df.dropna(axis=0, subset=['OS', 'OS.days', 'mean_tmb'], inplace=True)
io_drugs = 'Atezolizumab|Durvalumab|Ipilimumab|Nivolumab|Pembrolizumab'
df = df.loc[~df['regimen_drugs'].str.contains(io_drugs)]
print(len(df))

tmb = df['mean_tmb'].values
df = df.loc[tmb < np.percentile(tmb, 99)]
tmb = t.trf(df.mean_tmb.values)
times = df['OS.days'].values
events = df['OS'].values

test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig3' / ('nsclc_nonio.pkl'), 'rb'))

cox_losses = []
for idx_test in test_idx:
    mask = np.ones(len(df), dtype=bool)
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
    
