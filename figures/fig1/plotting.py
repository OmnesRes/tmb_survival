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
import seaborn as sns
import numpy as np
import pickle

t = utils.LogTransform(bias=4, min_x=0)

tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'linear_data.pkl', 'rb'))

x = np.linspace(min(tmb), max(tmb), 20)
risks = x * .5

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=1)
fig.subplots_adjust(left=.13)
fig.subplots_adjust(right=1)
ax.plot(x, risks, alpha=1, linewidth=2, color='k')
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='k')
ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 40, 64]])
ax.set_xticklabels([0, 2, 5, 10, 20, 40, 64])
ax.set_yticks([0, .5, 1, 1.5, 2])
ax.tick_params(axis='y', length=8, width=1, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(64))
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_bounds(0, 2)
ax.spines['left'].set_position(['outward', 5])
ax.spines['left'].set_linewidth(1)
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Log Partial Hazard', fontsize=12)
plt.savefig(cwd / 'figures' / 'fig1' / 'linear_risks.pdf')


tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'nonmonotonic_data.pkl', 'rb'))

x = np.linspace(min(tmb), max(tmb), 200)
risks = ((-(x - 2)**2) * 15 + 40) * .05

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=1)
fig.subplots_adjust(left=.13)
fig.subplots_adjust(right=1)
ax.plot(x, risks, alpha=1, linewidth=2, color='k')
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='k')
ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 40, 64]])
ax.set_xticklabels([0, 2, 5, 10, 20, 40, 64])
ax.set_yticks([-1.0, 0.0, 1.0, 2.0])
ax.set_yticklabels([-1.0, 0.0, 1.0, 2.0])
ax.tick_params(axis='y', length=8, width=1, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(64))
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_bounds(-1, 2)
ax.spines['left'].set_position(['outward', 5])
ax.spines['left'].set_linewidth(1)
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Log Partial Hazard', fontsize=12)
plt.savefig(cwd / 'figures' / 'fig1' / 'nonmonotonic_risks.pdf')



tmb, sim_risks, times_events = pickle.load(open(cwd / 'figures' / 'fig1' / 'step_data.pkl', 'rb'))

x = np.linspace(min(tmb), max(tmb), 200)
risks = [0 if i <= np.median(tmb) else 1 for i in x]

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.129)
fig.subplots_adjust(top=1)
fig.subplots_adjust(left=.13)
fig.subplots_adjust(right=1)
ax.plot(x, risks, alpha=1, linewidth=2, color='k')
sns.rugplot(data=tmb, ax=ax, alpha=.5, color='k')
ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 40, 64]])
ax.set_xticklabels([0, 2, 5, 10, 20, 40, 64])
ax.set_yticks([0.0, .5, 1.0])
ax.set_yticklabels([0.0, .5, 1.0])
ax.tick_params(axis='y', length=8, width=1, direction='out', labelsize=10)
ax.tick_params(axis='x', length=8, width=1, direction='out', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(64))
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_bounds(0, 1)
ax.spines['left'].set_position(['outward', 5])
ax.spines['left'].set_linewidth(1)
ax.set_xlabel('TMB', fontsize=12)
ax.set_ylabel('Log Partial Hazard', fontsize=12)
plt.savefig(cwd / 'figures' / 'fig1' / 'step_risks.pdf')


