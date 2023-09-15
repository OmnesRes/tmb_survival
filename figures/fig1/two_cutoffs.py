import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_survival':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_survival')]
    import sys
    sys.path.append(str(cwd))
from tqdm import tqdm
import concurrent.futures
import numpy as np
from lifelines.statistics import logrank_test
import pickle


# tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'linear_data.pkl', 'rb'))
# tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'nonmonotonic_data.pkl', 'rb'))
tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'step_data.pkl', 'rb'))


##need to sort
indexes = np.argsort(tmb)
tmb = np.sort(tmb)

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

    times = np.array(times)[indexes]
    events = np.array(events)[indexes]

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


with open(cwd / 'figures' / 'fig1' / 'two_cutoffs_linear.pkl', 'wb') as f:
    pickle.dump(best_indexes, f)

with open(cwd / 'figures' / 'fig1' / 'two_cutoffs_nonmonotonic.pkl', 'wb') as f:
    pickle.dump(best_indexes, f)

with open(cwd / 'figures' / 'fig1' / 'two_cutoffs_step.pkl', 'wb') as f:
    pickle.dump(best_indexes, f)
