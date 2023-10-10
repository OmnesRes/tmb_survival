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

tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'nonmonotonic_data.pkl', 'rb'))
indexes = np.argsort(tmb)
times = np.array([i[0][4] for i in times_events])
events = np.array([i[1][4] for i in times_events])

test_idx, results = pickle.load(open(cwd / 'figures' / 'fig2' / 'nonmonotonic_data_runs_4.pkl', 'rb'))

cph = CoxPHFitter()
for model in results:
    print(model)
    concordance = concordance_index(times[np.concatenate(test_idx)], np.concatenate(results[model][0]), events[np.concatenate(test_idx)])
    print(round(concordance, 3))
    losses = []
    for idx_test, risks in zip(test_idx, results[model][1]):
        mask = np.ones(len(risks), dtype=bool)
        mask[idx_test] = False
        cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
        losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
    print(round(np.mean(losses), 3))
    
###cox
cox_losses = []
for idx_test in test_idx:
    mask = np.ones(len(risks), dtype=bool)
    mask[idx_test] = False
    cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': tmb[mask]}), 'T', 'E', formula='x')
    cox_losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': tmb[idx_test]})))
print('cox')
print(round(cph.fit(pd.DataFrame({'T': times, 'E': events, 'x': tmb}), 'T', 'E', formula='x').concordance_index_, 3))
print(round(np.mean(cox_losses), 3))


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



