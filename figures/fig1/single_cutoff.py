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
from figures.sim_tools import *
from lifelines.statistics import logrank_test
import pickle


# tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'linear_data.pkl', 'rb'))
# tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'nonmonotonic_data.pkl', 'rb'))
tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'step_data.pkl', 'rb'))

best_index = {}
for choice in range(15):
    print(choice)
    times = [i[0][choice] for i in times_events]
    events = [i[1][choice] for i in times_events]

    times = np.array(times)
    events = np.array(events)

    stats = []
    offset = 25
    for index, cutoff in enumerate(tmb[offset: -offset]):
        stats.append(logrank_test(times[: index + offset + 1], times[index + offset + 1:], event_observed_A=events[: index + offset + 1], event_observed_B=events[index + offset + 1:]).test_statistic)

    cutoff = np.argmax(stats)
    best_index[choice] = cutoff



# with open(cwd / 'figures' / 'cutoffs' / 'sim' / 'single_cutoff_linear.pkl', 'wb') as f:
#     pickle.dump(best_index, f)
