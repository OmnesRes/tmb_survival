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
import seaborn as sns

labels_to_use = ['BLCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC']

data = pickle.load(open(cwd / 'files' / 'data.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
tmb_dict = {i[:12]: data[i][0] / (data[i][1] / 1e6) for i in data}

samples['tmb'] = samples.bcr_patient_barcode.apply(lambda x: tmb_dict.get(x, np.nan))
samples.dropna(axis=0, subset=['OS', 'OS.time', 'tmb'], inplace=True)

label_dict = {'FCN': 'FCN', 'sigmoid': 'Sigmoid', '2-neuron': '2 Neuron'}
t = utils.LogTransform(bias=4, min_x=0)
cph = CoxPHFitter()

for cancer in labels_to_use:
    print(cancer)
    df = samples.loc[samples['type'] == cancer]
    tmb = df['tmb'].values
    df = df.loc[tmb < np.percentile(tmb, 99)]
    tmb = t.trf(df.tmb.values)
    times = df['OS.time'].values
    events = df['OS'].values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=.129)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(left=.04)
    fig.subplots_adjust(right=1)
    cph.fit(pd.DataFrame({'T': times, 'E': events, 'x': tmb}), 'T', 'E', formula='x')
    ax.plot(tmb, tmb * cph.params_[0] - np.mean(tmb * cph.params_[0]), linewidth=2, alpha=.5, label='Cox')
    for model in ['FCN', '2neuron', 'sigmoid']:
        test_idx, test_ranks, all_risks = pickle.load(open(cwd / 'figures' / 'fig3' / (model + '_runs.pkl'), 'rb'))[cancer]
        normed_risks = []
        for idx_test, risks in zip(test_idx, all_risks):
            mask = np.ones(len(risks), dtype=bool)
            mask[idx_test] = False
            cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
            normed_risks.append(risks[:, 0] * cph.params_[0])
        overall_risks = np.mean([i - np.mean(i) for i in normed_risks], axis=0)
        indexes = np.argsort(tmb)
        ax.plot(np.sort(tmb), overall_risks[indexes], linewidth=2, alpha=.5, label=label_dict[model])
        
# ax.set_xticks(t.trf(np.array([0, 2, 5, 10, 20, 40, 64])))
# ax.set_xticklabels([0, 2, 5, 10, 20, 40, 64])
ax.set_yticks([])
ax.tick_params(axis='y', length=0, width=0, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
# ax.spines['bottom'].set_bounds(t.trf(0), t.trf(64))
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Log Partial Hazard', fontsize=12)
ax.set_title(cancer)
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='#1f77b4')
plt.legend(frameon=False)
plt.show()

# plt.savefig(cwd / 'figures' / 'fig2' / 'linear_data.pdf')




