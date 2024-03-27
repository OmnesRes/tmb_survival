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
import pandas as pd
import pickle
from model import utils
from matplotlib import pyplot as plt
import seaborn as sns

t = utils.LogTransform(bias=4, min_x=0)
cph = CoxPHFitter()

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
indexes = np.argsort(tmb)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=.95)
fig.subplots_adjust(left=.04)
fig.subplots_adjust(right=1)
###cox
cox_risks = []
for idx_test in test_idx:
    mask = np.ones(len(tmb), dtype=bool)
    mask[idx_test] = False
    cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
    cox_risks.append(tmb * cph.params_[0])
ax.plot(np.sort(tmb), np.mean([i - np.mean(i) for i in cox_risks], axis=0)[indexes], linewidth=2, alpha=.5, label='Cox', color='#2ca02c')
for model in ['FCN']:
    losses = []
    normed_risks = []
    for idx_test, risks in zip(test_idx, results[model][1]):
        mask = np.ones(len(risks), dtype=bool)
        mask[idx_test] = False
        cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
        losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
        normed_risks.append(risks[:, 0] * cph.params_[0])
    print(np.mean(losses))
    overall_risks = np.mean([i - np.mean(i) for i in normed_risks], axis=0)
    ax.plot(np.sort(tmb), overall_risks[indexes], linewidth=2, alpha=.5, label=model, color='#ff7f0e')
    
ax.set_xticks(t.trf(np.array([0, 2, 5, 10, 20, 40, 80, 140])))
ax.set_xticklabels([0, 2, 5, 10, 20, 40, 80, 140])
ax.set_yticks([])
ax.tick_params(axis='y', length=0, width=0, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(140))
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Normalized Log Partial Hazard', fontsize=12)
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='#1f77b4')
ax.set_title('Melanoma IO')
plt.legend(frameon=False, loc='upper center', ncol=4)
plt.savefig(cwd / 'figures' / 'supp_fig2' / 'melanoma_io.pdf')

df = pd.read_csv(cwd / 'figures' / 'supp_fig2'/ 'data' / '41588_2018_312_MOESM3_ESM.csv', sep=',', low_memory=False, skiprows=1)
df = df.loc[df['Cancer.Type'] == 'Non-Small Cell Lung Cancer']
sample_info = pd.read_csv(cwd / 'figures' / 'supp_fig2'/ 'data' / 'data_clinical_sample.txt', sep='\t', low_memory=False, skiprows=4)
df = pd.merge(df, sample_info, left_on='Sample.ID', right_on='SAMPLE_ID')

tmb = df['TMB_NONSYNONYMOUS'].values
df = df.loc[tmb < np.percentile(tmb, 99)]
tmb = t.trf(df.TMB_NONSYNONYMOUS.values)
times = df['SURVIVAL_MONTHS'].values
events = df['SURVIVAL_EVENT'].values

test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig2' / 'NSCLC_io.pkl', 'rb'))
indexes = np.argsort(tmb)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=.95)
fig.subplots_adjust(left=.04)
fig.subplots_adjust(right=1)
###cox
cox_risks = []
for idx_test in test_idx:
    mask = np.ones(len(tmb), dtype=bool)
    mask[idx_test] = False
    cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
    cox_risks.append(tmb * cph.params_[0])
ax.plot(np.sort(tmb), np.mean([i - np.mean(i) for i in cox_risks], axis=0)[indexes], linewidth=2, alpha=.5, label='Cox', color='#2ca02c')
for model in ['FCN']:
    losses = []
    normed_risks = []
    for idx_test, risks in zip(test_idx, results[model][1]):
        mask = np.ones(len(risks), dtype=bool)
        mask[idx_test] = False
        cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
        losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
        normed_risks.append(risks[:, 0] * cph.params_[0])
    print(np.mean(losses))
    overall_risks = np.mean([i - np.mean(i) for i in normed_risks], axis=0)
    ax.plot(np.sort(tmb), overall_risks[indexes], linewidth=2, alpha=.5, label=model, color='#ff7f0e')
    
ax.set_xticks(t.trf(np.array([0, 2, 5, 10, 20, 50])))
ax.set_xticklabels([0, 2, 5, 10, 20, 50])
ax.set_yticks([])
ax.tick_params(axis='y', length=0, width=0, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(50))
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Normalized Log Partial Hazard', fontsize=12)
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='#1f77b4')
ax.set_title('NSCLC IO (2019)')
plt.legend(frameon=False, loc='upper center', ncol=4)
plt.savefig(cwd / 'figures' / 'supp_fig2' / 'nsclc_io_2019.pdf')


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
indexes = np.argsort(tmb)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=.95)
fig.subplots_adjust(left=.04)
fig.subplots_adjust(right=1)
###cox
cox_risks = []
for idx_test in test_idx:
    mask = np.ones(len(tmb), dtype=bool)
    mask[idx_test] = False
    cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
    cox_risks.append(tmb * cph.params_[0])
ax.plot(np.sort(tmb), np.mean([i - np.mean(i) for i in cox_risks], axis=0)[indexes], linewidth=2, alpha=.5, label='Cox', color='#2ca02c')
for model in ['FCN']:
    losses = []
    normed_risks = []
    for idx_test, risks in zip(test_idx, results[model][1]):
        mask = np.ones(len(risks), dtype=bool)
        mask[idx_test] = False
        cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
        losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
        normed_risks.append(risks[:, 0] * cph.params_[0])
    print(np.mean(losses))
    overall_risks = np.mean([i - np.mean(i) for i in normed_risks], axis=0)
    ax.plot(np.sort(tmb), overall_risks[indexes], linewidth=2, alpha=.5, label=model, color='#ff7f0e')
    
ax.set_xticks(t.trf(np.array([0, 2, 5, 10, 20, 50])))
ax.set_xticklabels([0, 2, 5, 10, 20, 50])
ax.set_yticks([])
ax.tick_params(axis='y', length=0, width=0, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(50))
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Normalized Log Partial Hazard', fontsize=12)
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='#1f77b4')
ax.set_title('NSCLC IO (2021)')
plt.legend(frameon=False, loc='upper center', ncol=4)
plt.savefig(cwd / 'figures' / 'supp_fig2' / 'nsclc_io_2021.pdf')




