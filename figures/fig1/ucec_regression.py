import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_survival':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_survival')]
import sys
sys.path.append(str(cwd))
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import concurrent.futures
from model import utils
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
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
plt.savefig(cwd / 'figures' / 'fig1' / 'histogram.pdf')

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
plt.savefig(cwd / 'figures' / 'fig1' / 'survival_curves.pdf')


t = utils.LogTransform(bias=4, min_x=0)
tmb = t.trf(tmb)

bin = .1
counts = dict(zip(*np.unique(np.around(tmb, 1), return_counts=True)))
counts.update({round(i, 1): counts.get(round(i, 1), 0) for i in range(0, int(t.trf(1000) / 10))})
values = [sum([counts.get((i / 10) + j / 10, 0) for j in range(int(bin * 10))]) for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.108)
fig.subplots_adjust(top=1)
fig.subplots_adjust(left=.1)
fig.subplots_adjust(right=.97)
ax.bar([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))], values, width=bin, alpha=.3)
ax.set_xticks(t.trf(np.array([0, 2, 5, 10, 20, 40, 100, 200, 500, 1000])))
ax.set_xticklabels([0, 2, 5, 10, 20, 40, 100, 200, 500, 1000])
ax.set_yticks(list(range(0, 100, 10)))
ax.set_yticklabels(list(range(0, 100, 10)))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_bounds(0, t.trf(1000))
ax.spines['left'].set_bounds(0, 90)
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Counts', fontsize=12)
plt.savefig(cwd / 'figures' / 'fig1' / 'log_histogram.pdf')

cph = CoxPHFitter()
samples['tmb'] = tmb
cph.fit(samples, duration_col='OS.time', event_col='OS', formula="tmb")
x_times = np.linspace(0, 365 * 5, 200)
survival_probs = cph.predict_survival_function(t.trf(np.array([1, 10, 64, 256, 512]))[:, np.newaxis], x_times)

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
plt.savefig(cwd / 'figures' / 'fig1' / 'log_survival_curves.pdf')


##single cutoff
indexes = np.argsort(tmb)
tmb = np.sort(tmb)


times = samples['OS.time'].values[indexes]
events = samples['OS'].values[indexes]
stats = []
offset = 25
for index, cutoff in enumerate(tmb[offset: -offset]):
    stats.append(logrank_test(times[: index + offset + 1], times[index + offset + 1:], event_observed_A=events[: index + offset + 1], event_observed_B=events[index + offset + 1:]).test_statistic)

cutoff = np.argmax(stats)

samples['high'] = samples['tmb'].apply(lambda x: (x >= tmb[cutoff + offset + 1]).astype(np.int16))

def get_statistic(index):
    stats = []
    for index2 in range(len(tmb[offset + index: -(offset + offset2 - 1)])):
        stats.append(logrank_test(np.concatenate([times[: index + offset + 1], times[index + offset + offset2 + index2 + 1:]]),
                                  times[index + offset + 1: index + offset + offset2 + index2 + 1],
                                  event_observed_A=np.concatenate([events[: index + offset + 1], events[index + offset + offset2 + index2 + 1:]]),
                                  event_observed_B=events[index + offset + 1: index + offset + offset2 + index2 + 1]
                                  ).test_statistic)
    return stats


offset = 25
offset2 = 50

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for index, result in tqdm(zip(range(len(tmb[offset: -(offset + offset2 - 1)])), executor.map(get_statistic, range(len(tmb[offset: -(offset + offset2 - 1)]))))):
        stats[index] = result

max_stat = 0
for i in stats:
    if max(stats[i]) > max_stat:
        max_stat = max(stats[i])
        max_index = i

index = max_index
index2 = np.argmax(stats[max_index])

# index=15
# index2=255

samples['mid'] = samples['tmb'].apply(lambda x: ((x >= tmb[index + offset + 1]) and (x <= tmb[index + offset + offset2 + index2])).astype(np.int16))

kmf = KaplanMeierFitter()
fig = plt.figure()
fig.subplots_adjust(bottom=.11)
fig.subplots_adjust(top=.95)
fig.subplots_adjust(left=.115)
fig.subplots_adjust(right=.95)
ax = fig.add_subplot(111)
kmf.fit(samples.loc[samples['high'] == False]['OS.time'].values, samples.loc[samples['high'] == False]['OS'].values, label='Low TMB').plot_survival_function(ax=ax, show_censors=True, ci_alpha=0, censor_styles={"marker": "|", "ms":5}, color='#1f77b4')
kmf.fit(samples.loc[samples['high'] == True]['OS.time'].values, samples.loc[samples['high'] == True]['OS'].values, label='High TMB').plot_survival_function(ax=ax, show_censors=True, ci_alpha=0, censor_styles={"marker": "|", "ms":5}, color='#d62728')
kmf.fit(samples.loc[samples['mid'] == False]['OS.time'].values, samples.loc[samples['mid'] == False]['OS'].values, label='Moderate TMB').plot_survival_function(ax=ax, show_censors=True, ci_alpha=0, censor_styles={"marker": "|", "ms":5, "alpha":.5}, color='#d62728', alpha=.5)
kmf.fit(samples.loc[samples['mid'] == True]['OS.time'].values, samples.loc[samples['mid'] == True]['OS'].values, label='Extreme TMB').plot_survival_function(ax=ax, show_censors=True, ci_alpha=0, censor_styles={"marker": "|", "ms":5, "alpha":.5}, color='#1f77b4', alpha=.5)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='x', length=7, width=1, direction='out', labelsize=12)
ax.tick_params(axis='y', length=7, width=1, direction='out', labelsize=12)
ax.set_yticklabels([0, 20, 40, 60, 80, 100])
ax.set_xticks(np.arange(0, 8000, 1000))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_bounds(0, 7000)
ax.spines['left'].set_bounds(0, 1)
ax.set_xlabel('Days', fontsize=12)
ax.set_ylabel('% Surviving', fontsize=12)
plt.legend(frameon=False)
plt.savefig(cwd / 'figures' / 'fig1' / ('kaplan.pdf'))






