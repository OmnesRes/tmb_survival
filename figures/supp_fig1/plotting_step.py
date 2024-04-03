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

tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'supp_fig1' / 'step_data.pkl', 'rb'))
indexes = np.argsort(tmb)
times = np.array([i[0][0] for i in times_events])
events = np.array([i[1][0] for i in times_events])

test_idx, results = pickle.load(open(cwd / 'figures' / 'supp_fig1' / 'step_data_runs.pkl', 'rb'))

label_dict = {'FCN': 'FCN', 'sigmoid': 'Sigmoid'}


cph = CoxPHFitter()

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=.95)
fig.subplots_adjust(left=.04)
fig.subplots_adjust(right=1)
ax.plot(np.sort(tmb), (sim_risks - np.mean(sim_risks))[indexes], linewidth=2, label='True', color='k')
ax.scatter(tmb, sim_risks - np.mean(sim_risks), color='#1f77b4', alpha=.3)

###cox
cox_losses = []
cox_risks = []
for idx_test in test_idx:
    mask = np.ones(len(tmb), dtype=bool)
    mask[idx_test] = False
    cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
    cox_losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': tmb[idx_test]})))
    cox_risks.append(tmb * cph.params_[0])
print('cox')
print(round(cph.fit(pd.DataFrame({'T': times, 'E': events, 'x': tmb}), 'T', 'E', formula='x').concordance_index_, 3))
print(round(np.mean(cox_losses), 3))
ax.plot(np.sort(tmb), np.mean([i - np.mean(i) for i in cox_risks], axis=0)[indexes], linewidth=2, alpha=.5, label='Cox', color='#2ca02c')

for index, model in enumerate(['FCN', 'sigmoid']):
    losses = []
    normed_risks = []
    for idx_test, risks in zip(test_idx, results[model][1]):
        mask = np.ones(len(risks), dtype=bool)
        mask[idx_test] = False
        cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
        losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
        normed_risks.append(risks[:, 0] * cph.params_[0])
    print(np.mean(losses))
    concordance = concordance_index(times[np.concatenate(test_idx)], np.concatenate(results[model][0]), events[np.concatenate(test_idx)])
    print(round(concordance, 3))
    overall_risks = np.mean([i - np.mean(i) for i in normed_risks], axis=0)
    ax.plot(np.sort(tmb), overall_risks[indexes], linewidth=2, alpha=.5, label=label_dict[model], color=['#ff7f0e', '#9467bd'][index])
    
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
ax.set_ylim(-.75, 1)
ax.set_title('Step Data')
plt.legend(frameon=False, loc='upper center', ncol=5)
plt.savefig(cwd / 'figures' / 'supp_fig1' / 'step_data.pdf')

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


