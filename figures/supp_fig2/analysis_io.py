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

#from: https://www.nature.com/articles/s41588-018-0312-8
df = pd.read_csv(cwd / 'figures' / 'supp_fig2'/ 'data' / '41588_2018_312_MOESM3_ESM.csv', sep=',', low_memory=False, skiprows=1)
##limit to Melanoma
df = df.loc[df['Cancer.Type'] == 'Melanoma']
##remove uveal
sample_info = pd.read_csv(cwd / 'figures' / 'supp_fig2'/ 'data' / 'data_clinical_sample.txt', sep='\t', low_memory=False, skiprows=4)
df = pd.merge(df, sample_info, left_on='Sample.ID', right_on='SAMPLE_ID')
df = df.loc[~(df['CANCER_TYPE_DETAILED'] == 'Uveal Melanoma')]

tmb = df['TMB_NONSYNONYMOUS'].values
df = df.loc[tmb < np.percentile(tmb, 99)]
tmb = t.trf(df.TMB_NONSYNONYMOUS.values)
times = df['SURVIVAL_MONTHS'].values
events = df['SURVIVAL_EVENT'].values

test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig2' / 'SKCM_io.pkl', 'rb'))

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

#from: https://www.nature.com/articles/s41588-018-0312-8
df = pd.read_csv(cwd / 'figures' / 'supp_fig2'/ 'data' / '41588_2018_312_MOESM3_ESM.csv', sep=',', low_memory=False, skiprows=1)
##limit to NSCLC
df = df.loc[df['Cancer.Type'] == 'Non-Small Cell Lung Cancer']

sample_info = pd.read_csv(cwd / 'figures' / 'supp_fig2'/ 'data' / 'data_clinical_sample.txt', sep='\t', low_memory=False, skiprows=4)
df = pd.merge(df, sample_info, left_on='Sample.ID', right_on='SAMPLE_ID')

tmb = df['TMB_NONSYNONYMOUS'].values
df = df.loc[tmb < np.percentile(tmb, 99)]
tmb = t.trf(df.TMB_NONSYNONYMOUS.values)
times = df['SURVIVAL_MONTHS'].values
events = df['SURVIVAL_EVENT'].values

test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig2' / 'NSCLC_io.pkl', 'rb'))

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


#from: https://www.nature.com/articles/s41588-020-00752-4, https://zenodo.org/record/4074184
df = pd.read_csv(cwd / 'figures' / 'supp_fig2'/ 'data' / 'TMB_public.csv', sep=',', low_memory=False)
df = df.loc[(~(df['Type of ICI treatment'] == 'Never received ICI'))]
df['OS'] = (df['Vital status'] == 'DECEASED').astype(np.int32)

df = df.loc[df['Cancer type'] == 'NSCLC']
tmb = df['TMB (mutations/Mb)'].values
df = df.loc[tmb < np.percentile(tmb, 99)]
tmb = t.trf(df['TMB (mutations/Mb)'].values)
times = df['Overall Survival from diagnosis (Months)'].values
events = df['OS'].values

test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig2' / 'NSCLC_io_cohort2.pkl', 'rb'))

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


