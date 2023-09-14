from matplotlib import pyplot as plt
from lifelines import KaplanMeierFitter
from figures.sim_tools import *
# import pandas as pd
from tqdm import tqdm
import concurrent.futures
from lifelines.statistics import logrank_test
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_surv':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_surv')]
    import sys
    sys.path.append(str(cwd))

# tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'cutoffs' / 'sim' / 'linear_data.pkl', 'rb'))
# tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'cutoffs' / 'sim' / 'nonmonotonic_data.pkl', 'rb'))

def get_statistic(index):
    stats = []
    for index2 in range(len(tmb[offset + index: -(offset + offset2 - 1)])):
        stats.append(logrank_test(np.concatenate([times[: index + offset + 1], times[index + offset + offset2 + index2 + 1:]]),
                                  times[index + offset + 1: index + offset + offset2 + index2 + 1],
                                  event_observed_A=np.concatenate([events[: index + offset + 1], events[index + offset + offset2 + index2 + 1:]]),
                                  event_observed_B=events[index + offset + 1: index + offset + offset2 + index2 + 1]
                                  ).test_statistic)
    return stats

best_indexes = {}
for choice in range(15):
    times = [i[0][choice] for i in times_events]
    events = [i[1][choice] for i in times_events]

    times = np.array(times)
    events = np.array(events)

    stats = {}
    offset = 25
    offset2 = 50

    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        for index, result in tqdm(zip(range(len(tmb[offset: -(offset + offset2 - 1)])), executor.map(get_statistic, range(len(tmb[offset: -(offset + offset2 - 1)]))))):
            stats[index] = result

    max_stat = 0
    for i in stats:
        if max(stats[i]) > max_stat:
            max_stat = max(stats[i])
            max_index = i

    index = max_index
    index2 = np.argmax(stats[max_index])
    best_indexes[choice] = [index, index2]


# with open(cwd / 'figures' / 'cutoffs' / 'sim' / 'two_cutoffs_linear.pkl', 'wb') as f:
#     pickle.dump(best_indexes, f)

# with open(cwd / 'figures' / 'cutoffs' / 'sim' / 'two_cutoffs_nonmonotonic.pkl', 'wb') as f:
#     pickle.dump(best_indexes, f)
