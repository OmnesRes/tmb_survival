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

tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig2' / 'linear_data.pkl', 'rb'))
times = np.array([i[0][0] for i in times_events])
events = np.array([i[1][0] for i in times_events])

test_idx, results = pickle.load(open(cwd / 'figures' / 'fig3' / 'linear_data_runs_0.pkl', 'rb'))

cph = CoxPHFitter()

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=.95)
fig.subplots_adjust(left=.04)
fig.subplots_adjust(right=1)
ax.plot(tmb, sim_risks - np.mean(sim_risks), linewidth=2, label='True', color='k')
ax.scatter(tmb, sim_risks - np.mean(sim_risks), color='#1f77b4', alpha=.3)

for model in ['FCN']:
    losses = []
    for index, (idx_test, risks) in enumerate(zip(test_idx, results[model][1])):
        mask = np.ones(len(risks), dtype=bool)
        mask[idx_test] = False
        cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
        losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
        normed_risks = (risks[:, 0] * cph.params_[0])
        ax.plot(tmb, normed_risks - np.mean(normed_risks), linewidth=2, alpha=.5, label='Fold' + str(index + 1))

ax.set_xticks(t.trf(np.array([0, 2, 5, 10, 20, 40, 64])))
ax.set_xticklabels([0, 2, 5, 10, 20, 40, 64])
ax.set_yticks([])
ax.tick_params(axis='y', length=0, width=0, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(64))
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Normalized Log Partial Hazard', fontsize=12)
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='#1f77b4')
ax.set_title('Linear Data')
plt.legend(frameon=False, loc='upper center', ncol=5)
plt.savefig(cwd / 'figures' / 'supp_fig2' / 'linear_data_0.pdf')
