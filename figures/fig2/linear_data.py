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
from lifelines import KaplanMeierFitter
from figures.sim_tools import *
import pickle
import pandas as pd

data = pickle.load(open(cwd / 'files' / 'data.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
tmb_dict = {i[:12]: data[i][0] / (data[i][1] / 1e6) for i in data}

samples['tmb'] = samples.bcr_patient_barcode.apply(lambda x: tmb_dict.get(x, np.nan))
samples.dropna(axis=0, subset=['tmb'], inplace=True)

tmb = samples.loc[samples['type'] == "UCEC"]['tmb'].values
tmb = np.sort(tmb)
mask = tmb < 64
tmb = tmb[mask]

t = utils.LogTransform(bias=4, min_x=0)
tmb = t.trf(tmb)

beta = .5
sim_risks = tmb * beta
times_events = [generate_times(risk=i) for i in sim_risks]

with open(cwd / 'figures' / 'fig2' / 'linear_data.pkl', 'wb') as f:
    pickle.dump([tmb, sim_risks, times_events], f)
