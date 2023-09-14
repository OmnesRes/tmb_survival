import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_survival':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_survival')]
import sys
sys.path.append(str(cwd))
from matplotlib import pyplot as plt
from lifelines import KaplanMeierFitter
import numpy as np
from lifelines import CoxPHFitter
import pandas as pd
from lifelines.statistics import logrank_test
from model import utils
import pickle


data = pickle.load(open(cwd / 'files' / 'data.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))
samples = samples.loc[samples['type'] == "UCEC"]
[data.pop(i) for i in list(data.keys()) if not data[i]]
tmb_dict = {i[:12]: data[i][0] / (data[i][1] / 1e6) for i in data}
samples['tmb'] = samples.bcr_patient_barcode.apply(lambda x: tmb_dict.get(x, np.nan))
samples.dropna(axis=0, subset=['OS', 'OS.time', 'tmb'], inplace=True)

##tmb histogram
tmb = samples['tmb'].values

bin = 10
counts = dict(zip(*np.unique(np.around(tmb, 0), return_counts=True)))
counts.update({round(i, 0): counts.get(round(i, 0), 0) for i in range(0, 900)})
values = [sum([counts.get(i + j, 0) for j in range(bin)]) for i in range(int(min(counts.keys())), int(max(counts.keys()) + 1), bin)]

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.108)
fig.subplots_adjust(top=1)
fig.subplots_adjust(left=.1)
fig.subplots_adjust(right=.97)
ax.bar([i for i in range(int(min(counts.keys())), int(max(counts.keys()) + 1), bin)], np.log2(np.array(values) + 1), width=bin, alpha=.3)
ax.set_yticks(np.log2(np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256]) + 1))
ax.set_xticks([0, 200, 400, 600, 800, 1000])
ax.set_yticklabels([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_bounds(0, 1000)
ax.spines['left'].set_bounds(np.log2(1), np.log2(257))
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Counts', fontsize=12)
plt.savefig(cwd / 'figures' / 'supp_fig1' / 'histogram.pdf')

cph = CoxPHFitter()
cph.fit(samples, duration_col='OS.time', event_col='OS', formula="tmb")
x_times = np.linspace(0, 365 * 5, 200)
survival_probs = cph.predict_survival_function(np.array([1, 10, 64, 256, 512])[:, np.newaxis], x_times)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98)
fig.subplots_adjust(right=.99)
for index, tmb in enumerate([1, 10, 64, 256, 512]):
    ax.plot(x_times, survival_probs.values[:, index], label=tmb)
ax.set_ylim(.7, 1.001)
ax.set_xlim(0, 365 * 5)
ax.set_xticks([i * 365 for i in range(6)])
ax.set_xticklabels([0, 1, 2, 3, 4, 5])
ax.set_xlabel('Years', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(['outward', 5])
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['left'].set_bounds(.7, 1)
ax.legend(frameon=False, title='TMB', ncol=5)
plt.savefig(cwd / 'figures' / 'supp_fig1' / 'survival_curves.pdf')
