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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

labels_to_use = ['BLCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC']

data = pickle.load(open(cwd / 'files' / 'data.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
tmb_dict = {i[:12]: data[i][0] / (data[i][1] / 1e6) for i in data}

samples['tmb'] = samples.bcr_patient_barcode.apply(lambda x: tmb_dict.get(x, np.nan))
samples.dropna(axis=0, subset=['OS', 'OS.time', 'tmb'], inplace=True)

t = utils.LogTransform(bias=4, min_x=0)

model_stats = []
tmb_stats = []
best_models = []
model_labels = []
tmb_labels = []

cph = CoxPHFitter()
for cancer in labels_to_use:
    print(cancer)
    df = samples.loc[samples['type'] == cancer]
    tmb = df['tmb'].values
    df = df.loc[tmb < np.percentile(tmb, 99)]
    tmb = t.trf(df.tmb.values)
    times = df['OS.time'].values
    events = df['OS'].values
    model_losses = []
    for model in ['FCN', '2neuron', 'sigmoid']:
        test_idx, test_ranks, all_risks = pickle.load(open(cwd / 'figures' / 'fig3' / (model + '_runs.pkl'), 'rb'))[cancer]
        losses = []
        for idx_test, risks in zip(test_idx, all_risks):
            mask = np.ones(len(risks), dtype=bool)
            mask[idx_test] = False
            cph.fit(pd.DataFrame({'T': times[mask], 'E': events[mask], 'x': risks[mask][:, 0]}), 'T', 'E', formula='x')
            losses.append(cph.score(pd.DataFrame({'T': times[idx_test], 'E': events[idx_test], 'x': risks[idx_test][:, 0]})))
        model_losses.append(np.mean(losses))
    best_model = ['FCN', '2neuron', 'sigmoid'][np.argmax(model_losses)]
    best_models.append(best_model)
    test_idx, test_ranks, all_risks = pickle.load(open(cwd / 'figures' / 'fig3' / (best_model + '_runs.pkl'), 'rb'))[cancer]
    labels = []
    for idx_test, risks in zip(test_idx, all_risks):
        mask = np.ones(len(risks), dtype=bool)
        mask[idx_test] = False
        cutoff = np.median(risks[mask][:, 0])
        labels.append(risks[idx_test][:, 0] > cutoff)
    model_labels.append(np.concatenate(labels).astype(np.int32))
    
    cph.fit(pd.DataFrame(data={'x': np.concatenate(labels).astype(np.int32), 'OS.time': times[np.concatenate(test_idx)], 'OS': events[np.concatenate(test_idx)]}), duration_col='OS.time', event_col='OS', formula="x")
    model_stats.append([cph.summary['coef']['x'], cph.summary['se(coef)']['x'], cph.log_likelihood_ratio_test().summary.test_statistic.values[0]])
    
    ##cox model
    labels = []
    for idx_test, risks in zip(test_idx, all_risks):
        mask = np.ones(len(risks), dtype=bool)
        mask[idx_test] = False
        cutoff = np.median(tmb[mask])
        labels.append(tmb[idx_test] > cutoff)
    tmb_labels.append(np.concatenate(labels).astype(np.int32))
    cph.fit(pd.DataFrame(data={'x': np.concatenate(labels).astype(np.int32), 'OS.time': times[np.concatenate(test_idx)], 'OS': events[np.concatenate(test_idx)]}), duration_col='OS.time', event_col='OS', formula="x")
    tmb_stats.append([cph.summary['coef']['x'], cph.summary['se(coef)']['x'], cph.log_likelihood_ratio_test().summary.test_statistic.values[0]])


y_indexes = list(range(len(model_stats)))[::-1]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.12)
fig.subplots_adjust(top=.94)
fig.subplots_adjust(left=.087)
fig.subplots_adjust(right=.74)
fig.subplots_adjust(wspace=0)

for index, i in enumerate(model_stats):
    ax.hlines(y_indexes[index], np.log2(np.exp(i[0] - 1.96 * i[1])), np.log2(np.exp(i[0] + 1.96 * i[1])))
    ax.vlines(np.log2(np.exp(i[0] - 1.96 * i[1])), y_indexes[index] - .2, y_indexes[index] + .2)
    ax.vlines(np.log2(np.exp(i[0] + 1.96 * i[1])), y_indexes[index] - .2, y_indexes[index] + .2)
    ax.scatter(np.log2(np.exp(i[0])), y_indexes[index], color='#1f77b4', s=50)
ax.vlines(np.log2(1), -1, max(y_indexes) + .5, color='k', alpha=.3, zorder=-100, linewidths=1)
ax.set_xticks(np.log2(np.array([.125, .25, .5, 1, 2, 4, 8])))
ax.set_xticklabels([.125, .25, .5, 1.0, 2.0, 4.0, 8.0])
ax.set_yticks(y_indexes)
ax.set_yticklabels(labels_to_use)
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

for index, i in enumerate(model_stats):
    ax.text(1, 1 - ((index + 1) / len(y_indexes)) + .03, round(i[2], 1), transform=ax.transAxes)
# plt.savefig(cwd / 'figures' / 'fig5' / 'model_hazards.pdf')
plt.show()


y_indexes = list(range(len(tmb_stats)))[::-1]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.12)
fig.subplots_adjust(top=.94)
fig.subplots_adjust(left=.087)
fig.subplots_adjust(right=.74)
fig.subplots_adjust(wspace=0)

for index, i in enumerate(tmb_stats):
    ax.hlines(y_indexes[index], np.log2(np.exp(i[0] - 1.96 * i[1])), np.log2(np.exp(i[0] + 1.96 * i[1])))
    ax.vlines(np.log2(np.exp(i[0] - 1.96 * i[1])), y_indexes[index] - .2, y_indexes[index] + .2)
    ax.vlines(np.log2(np.exp(i[0] + 1.96 * i[1])), y_indexes[index] - .2, y_indexes[index] + .2)
    ax.scatter(np.log2(np.exp(i[0])), y_indexes[index], color='#1f77b4', s=50)
ax.vlines(np.log2(1), -1, max(y_indexes) + .5, color='k', alpha=.3, zorder=-100, linewidths=1)
ax.set_xticks(np.log2(np.array([.125, .25, .5, 1, 2, 4, 8])))
ax.set_xticklabels([.125, .25, .5, 1.0, 2.0, 4.0, 8.0])
ax.set_yticks(y_indexes)
ax.set_yticklabels(labels_to_use)
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

for index, i in enumerate(tmb_stats):
    ax.text(1, 1 - ((index + 1) / len(y_indexes)) + .03, round(i[2], 1), transform=ax.transAxes)
    
# plt.savefig(cwd / 'figures' / 'fig5' / 'tmb_hazards.pdf')
plt.show()

##kaplans

for index, cancer in enumerate(['BLCA', 'KIRC', 'SARC', 'SKCM']):
    df = samples.loc[samples['type'] == cancer]
    tmb = df['tmb'].values
    df = df.loc[tmb < np.percentile(tmb, 99)]
    tmb = t.trf(df.tmb.values)
    times = df['OS.time'].values
    events = df['OS'].values
    test_idx, test_ranks, all_risks = pickle.load(open(cwd / 'figures' / 'fig3' / (best_model + '_runs.pkl'), 'rb'))[cancer]

    low_times = times[np.concatenate(test_idx)][model_labels[labels_to_use.index(cancer)] == 0]
    low_events = events[np.concatenate(test_idx)][model_labels[labels_to_use.index(cancer)] == 0]
    high_times = times[np.concatenate(test_idx)][model_labels[labels_to_use.index(cancer)] == 1]
    high_events = events[np.concatenate(test_idx)][model_labels[labels_to_use.index(cancer)] == 1]

    p_value = logrank_test(low_times, high_times, event_observed_A=low_events, event_observed_B=high_events).p_value

    fig = plt.figure()
    fig.subplots_adjust(bottom=.11)
    fig.subplots_adjust(top=.98)
    fig.subplots_adjust(left=.099)
    fig.subplots_adjust(right=.969)
    kmf = KaplanMeierFitter()
    ax = fig.add_subplot(111)
    kmf.fit(low_times, low_events, label='Low Risk').plot_survival_function(ax=ax, show_censors=True, ci_alpha=0, censor_styles={"marker": "|", "ms":5})
    kmf.fit(high_times, high_events, label='High Risk').plot_survival_function(ax=ax, show_censors=True, ci_alpha=0, censor_styles={"marker": "|", "ms":5})
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', length=7, width=1, direction='out', labelsize=12)
    ax.tick_params(axis='y', length=7, width=1, direction='out', labelsize=12)
    ax.set_xticks([np.arange(0, 6000, 1000), np.arange(0, 6000, 1000), np.arange(0, 7000, 1000), np.arange(0, 14000, 2000)][index])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_bounds(0, [5000, 5000, 6000, 12000][index])
    ax.spines['left'].set_bounds(0, 1)
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('% Surviving', fontsize=12)
    plt.legend(frameon=False)
    ax.text(.0, .97, 'Logrank p-value=%.1E' % p_value, transform=ax.transAxes)
    plt.savefig(cwd / 'figures' / 'fig5' / (cancer + '_model.pdf'))


    low_times = times[np.concatenate(test_idx)][tmb_labels[labels_to_use.index(cancer)] == 0]
    low_events = events[np.concatenate(test_idx)][tmb_labels[labels_to_use.index(cancer)] == 0]
    high_times = times[np.concatenate(test_idx)][tmb_labels[labels_to_use.index(cancer)] == 1]
    high_events = events[np.concatenate(test_idx)][tmb_labels[labels_to_use.index(cancer)] == 1]

    p_value = logrank_test(low_times, high_times, event_observed_A=low_events, event_observed_B=high_events).p_value

    fig = plt.figure()
    fig.subplots_adjust(bottom=.11)
    fig.subplots_adjust(top=.98)
    fig.subplots_adjust(left=.099)
    fig.subplots_adjust(right=.969)
    kmf = KaplanMeierFitter()
    ax = fig.add_subplot(111)
    kmf.fit(low_times, low_events, label='Low TMB').plot_survival_function(ax=ax, show_censors=True, ci_alpha=0, censor_styles={"marker": "|", "ms":5})
    kmf.fit(high_times, high_events, label='High TMB').plot_survival_function(ax=ax, show_censors=True, ci_alpha=0, censor_styles={"marker": "|", "ms":5})
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', length=7, width=1, direction='out', labelsize=12)
    ax.tick_params(axis='y', length=7, width=1, direction='out', labelsize=12)
    ax.set_xticks([np.arange(0, 6000, 1000), np.arange(0, 6000, 1000), np.arange(0, 7000, 1000), np.arange(0, 14000, 2000)][index])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_bounds(0, [5000, 5000, 6000, 12000][index])
    ax.spines['left'].set_bounds(0, 1)
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('% Surviving', fontsize=12)
    plt.legend(frameon=False)
    ax.text(.0, .97, 'Logrank p-value=%.1E' % p_value, transform=ax.transAxes)
    plt.savefig(cwd / 'figures' / 'fig5' / (cancer + '_tmb.pdf'))










