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

t = utils.LogTransform(bias=4, min_x=0)
cph = CoxPHFitter()

##io data
df = pickle.load(open(cwd / 'figures' / 'supp_fig2' / 'crc_sample_table.pkl', 'rb'))
df.dropna(axis=0, subset=['OS', 'OS.days', 'mean_tmb'], inplace=True)
io_drugs = 'Atezolizumab|Durvalumab|Ipilimumab|Nivolumab|Pembrolizumab'
df = df.loc[~df['regimen_drugs'].str.contains(io_drugs)]
tmb = df['mean_tmb'].values
df = df.loc[tmb < np.percentile(tmb, 99)]
tmb = t.trf(df.mean_tmb.values)
times = df['OS.days'].values
events = df['OS'].values

test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig2' / ('crc_nonio.pkl'), 'rb'))
indexes = np.argsort(tmb)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=.95)
fig.subplots_adjust(left=.04)
fig.subplots_adjust(right=.98)
###cox
cox_risks = []
for idx_test in test_idx:
    mask = np.ones(len(tmb), dtype=bool)
    mask[idx_test] = False
    cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
    cox_risks.append(tmb * cph.params_[0])
ax.plot(np.sort(tmb), np.mean([i - np.mean(i) for i in cox_risks], axis=0)[indexes], linewidth=2, alpha=.5, label='Cox')
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
    ax.plot(np.sort(tmb), overall_risks[indexes], linewidth=2, alpha=.5, label=model)
    
ax.set_xticks(t.trf(np.array([0, 2, 5, 10, 20, 50, 100])))
ax.set_xticklabels([0, 2, 5, 10, 20, 50, 100])
ax.set_yticks([])
ax.tick_params(axis='y', length=0, width=0, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(100))
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Log Partial Hazard', fontsize=12)
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='k')
ax.set_title('Colorectal NonIO data')
plt.legend(frameon=False, loc='upper center', ncol=4)
plt.show()
plt.savefig(cwd / 'figures' / 'supp_fig2' / 'crc_nonio_data.pdf')