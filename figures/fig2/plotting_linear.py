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

tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'linear_data.pkl', 'rb'))
times = np.array([i[0][4] for i in times_events])
events = np.array([i[1][4] for i in times_events])

test_idx, results = pickle.load(open(cwd / 'figures' / 'fig2' / 'linear_data_runs_4.pkl', 'rb'))


cph = CoxPHFitter()

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=.95)
fig.subplots_adjust(left=.04)
fig.subplots_adjust(right=1)
ax.plot(tmb, sim_risks - np.mean(sim_risks), linewidth=2, label='True', color='k')
cph.fit(pd.DataFrame({'T': times, 'E': events, 'x': tmb}), 'T', 'E', formula='x')
ax.plot(tmb, tmb * cph.params_[0] - np.mean(tmb * cph.params_[0]), linewidth=2, alpha=.5, label='Cox')
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
    ax.plot(np.sort(tmb), overall_risks, linewidth=2, alpha=.5, label=model)
    
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
ax.set_ylabel('Log Partial Hazard', fontsize=12)
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='k')
ax.set_title('Linear Data')
plt.legend(frameon=False, loc='upper center', ncol=5)
plt.savefig(cwd / 'figures' / 'fig2' / 'linear_data_4.pdf')




# ###cox
# cox_losses = []
# for idx_test in test_idx:
#     mask = np.ones(len(risks), dtype=bool)
#     mask[idx_test] = False
#     cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
#     cox_losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': tmb[idx_test]})))
# print(np.mean(cox_losses))

# print(cph.fit(pd.DataFrame({'T': times, 'E': events, 'x': tmb}), 'T', 'E', formula='x').concordance_index_)


# ###true values
# sim_risks = np.array(sim_risks)
# print(concordance_index(times, -sim_risks, events))
# true_losses = []
# for idx_test in test_idx:
#     mask = np.ones(len(risks), dtype=bool)
#     mask[idx_test] = False
#     cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': sim_risks[mask]}), 'T', 'E', formula='x')
#     true_losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': sim_risks[idx_test]})))
# print(np.mean(true_losses))



