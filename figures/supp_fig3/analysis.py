import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_survival':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_survival')]
import sys
sys.path.append(str(cwd))
from model import utils
from matplotlib import pyplot as plt
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
import pickle


single_cutoff = pickle.load(open(cwd / 'figures' / 'supp_fig3' / 'single_cutoff_step.pkl', 'rb'))
tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'supp_fig3' / 'step_data.pkl', 'rb'))


indexes = np.argsort(tmb)
tmb = np.sort(tmb)
t = utils.LogTransform(bias=4, min_x=0)

cph = CoxPHFitter()


single = []
for choice in range(15):
    print(choice)
    times = [i[0][choice] for i in times_events]
    events = [i[1][choice] for i in times_events]
    
    times = np.array(times)[indexes]
    events = np.array(events)[indexes]
    
    temp_df = pd.DataFrame(data={'tmb': tmb, 'OS.time': times, 'OS': events})
    offset = 25
    index = single_cutoff[choice]
    temp_df['high'] = temp_df['tmb'].apply(lambda x: (x >= tmb[index + offset + 1]).astype(np.int16))
    cph.fit(temp_df, duration_col='OS.time', event_col='OS', formula="high", fit_options={'step_size': .1})
    single.append([cph.summary['coef']['high'], cph.summary['se(coef)']['high'], cph.log_likelihood_ratio_test().summary.test_statistic.values[0], tmb[index + offset + 1]])


labels_to_use = list(range(1, 16))
y_indexes = list(range(len(single)))[::-1]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.12)
fig.subplots_adjust(top=.94)
fig.subplots_adjust(left=.087)
fig.subplots_adjust(right=.74)
fig.subplots_adjust(wspace=0)

for index, i in enumerate(single):
    ax.hlines(y_indexes[index], np.log2(np.exp(i[0] - 1.96 * i[1])), np.log2(np.exp(i[0] + 1.96 * i[1])))
    ax.vlines(np.log2(np.exp(i[0] - 1.96 * i[1])), y_indexes[index] - .2, y_indexes[index] + .2)
    ax.vlines(np.log2(np.exp(i[0] + 1.96 * i[1])), y_indexes[index] - .2, y_indexes[index] + .2)
    ax.scatter(np.log2(np.exp(i[0])), y_indexes[index], color='#1f77b4', s=50)
ax.vlines(np.log2(1), -1, max(y_indexes) + .5, color='k', alpha=.3, zorder=-100, linewidths=1)
ax.set_xticks(np.log2(np.array([.125, .25, .5, 1, 2, 4, 8])))
ax.set_xticklabels([.125, .25, .5, 1.0, 2.0, 4.0, 8.0])
ax.set_yticks(y_indexes)
ax.set_yticklabels([])
ax.tick_params(axis='y', length=0, labelsize=10, pad=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_bounds(np.log2(0.125), np.log2(8))
ax.spines['bottom'].set_position(['outward', -1])
ax.spines['bottom'].set_linewidth(1)
ax.set_ylim(-1, max(y_indexes) + .25)
ax.set_xlim(np.log2(0.0625), np.log2(16))
ax.text(1, 1.03, 'LL-test', transform=ax.transAxes)
ax.text(1.15, 1.03, '<= TMB', transform=ax.transAxes)

for index, i in enumerate(single):
    ax.text(1, 1 - ((index + 1) / len(y_indexes)) + .03, round(i[2], 1), transform=ax.transAxes)
    ax.text(1.3, 1 - ((index + 1) / len(y_indexes)) + .03, round(t.inv(i[3]), 2), transform=ax.transAxes)

plt.savefig(cwd / 'figures' / 'supp_fig3' / 'step_single.pdf')
