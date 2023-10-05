import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_survival':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_survival')]
import sys
sys.path.append(str(cwd))
import pickle
import pandas as pd
import numpy as np
from model.model import Encoders, NN
from model.layers import Losses, Activations
from model import utils
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

t = utils.LogTransform(bias=4, min_x=0)

df = pd.read_csv(cwd / 'figures' / 'msk_figure'/ 'data' / '41588_2018_312_MOESM3_ESM.csv', sep=',', low_memory=False, skiprows=1)
##limit to Melanoma
df = df.loc[df['Cancer.Type'] == 'Melanoma']
##remove uveal
sample_info = pd.read_csv(cwd / 'figures' / 'msk_figure'/ 'data' / 'data_clinical_sample.txt', sep='\t', low_memory=False, skiprows=4)
df = pd.merge(df, sample_info, left_on='Sample.ID', right_on='SAMPLE_ID')
df = df.loc[~(df['CANCER_TYPE_DETAILED'] == 'Uveal Melanoma')]

tmb = df['TMB_NONSYNONYMOUS'].values
df = df.loc[tmb < np.percentile(tmb, 99)]
tmb = t.trf(df.TMB_NONSYNONYMOUS.values)[:, np.newaxis]
times = df['SURVIVAL_MONTHS'].values[:, np.newaxis]
events = df['SURVIVAL_EVENT'].values[:, np.newaxis]
cancer_strat = np.zeros_like(df['TMB_NONSYNONYMOUS']) ##no cancer info
y_label = np.stack(np.concatenate([times, events, cancer_strat[:, np.newaxis]], axis=-1))
y_strat = (tmb[:, 0] > np.percentile(tmb[:, 0], 80)).astype(np.int32)


ds_all = tf.data.Dataset.from_tensor_slices((
                                            (
                                                tmb
                                            ),
                                            (
                                                y_label,
                                            ),
                                            ))
ds_all = ds_all.batch(len(y_label), drop_remainder=False)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=30, mode='min', restore_best_weights=True)]

results = {}

##FCN model
test_ranks = []
test_idx = []
all_risks = []
for idx_train, idx_test in StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(y_strat, y_strat):
    test_idx.append(idx_test)
    ds_train = tf.data.Dataset.from_tensor_slices((
        (
            tmb[idx_train]
        ),
        (
            y_label[idx_train],
        ),
    ))
    ds_train = ds_train.batch(len(idx_train), drop_remainder=False).repeat()
    temp_concordance = 0
    runs = 0
    while runs < 3:
        encoder_1 = Encoders.Encoder(shape=(1,), layers=(128, 128), dropout=.05)
        net = NN(encoders=[encoder_1.model], layers=(), norm=True)
        net.model.compile(loss=Losses.CoxPH(cancers=1),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        net.model.fit(ds_train, callbacks=callbacks, epochs=500, steps_per_epoch=10)
        risks = net.model.predict(ds_train, steps=1)
        temp_all_risks = net.model.predict(ds_all)
        try:
            concordance = concordance_index(times[idx_train], np.exp(-risks[:, 0]), events[idx_train])
            runs += 1
        except:
            concordance = 0
        if concordance > temp_concordance:
            temp_concordance = concordance
            best_all_risks = temp_all_risks
            temp = np.exp(-temp_all_risks[:, 0]).argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp))
    test_ranks.append(ranks[idx_test])
    all_risks.append(best_all_risks)
    
results['FCN'] = [test_ranks, all_risks]

##sigmoid model
test_ranks = []
all_risks = []
for idx_train, idx_test in StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(y_strat, y_strat):
    ds_train = tf.data.Dataset.from_tensor_slices((
        (
            tmb[idx_train]
        ),
        (
            y_label[idx_train],
        ),
    ))
    ds_train = ds_train.batch(len(idx_train), drop_remainder=False).repeat()
    temp_concordance = 0
    runs = 0
    while runs < 5:
        encoder_1 = Encoders.Encoder(shape=(1,), layers=(), dropout=0)
        net = NN(encoders=[encoder_1.model], layers=(1,), norm=False, default_activation=Activations.ASU())
        net.model.compile(loss=Losses.CoxPH(cancers=1),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        net.model.fit(ds_train, callbacks=callbacks, epochs=500, steps_per_epoch=10)
        risks = net.model.predict(ds_train, steps=1)
        temp_all_risks = net.model.predict(ds_all)
        try:
            concordance = concordance_index(times[idx_train], np.exp(-risks[:, 0]), events[idx_train])
            runs += 1
        except:
            concordance = 0
        if concordance > temp_concordance:
            temp_concordance = concordance
            best_all_risks = temp_all_risks
            temp = np.exp(-temp_all_risks[:, 0]).argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp))
    test_ranks.append(ranks[idx_test])
    all_risks.append(best_all_risks)

results['sigmoid'] = [test_ranks, all_risks]

##two neuron
test_ranks = []
all_risks = []
for idx_train, idx_test in StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(y_strat, y_strat):
    ds_train = tf.data.Dataset.from_tensor_slices((
        (
            tmb[idx_train]
        ),
        (
            y_label[idx_train],
        ),
    ))
    ds_train = ds_train.batch(len(idx_train), drop_remainder=False).repeat()
    temp_concordance = 0
    runs = 0
    while runs < 3:
        encoder_1 = Encoders.Encoder(shape=(1,), layers=(2,), dropout=0)
        net = NN(encoders=[encoder_1.model], layers=(), norm=False)
        net.model.compile(loss=Losses.CoxPH(cancers=1),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        net.model.fit(ds_train, callbacks=callbacks, epochs=1000, steps_per_epoch=10)
        net.model.compile(loss=Losses.CoxPH(cancers=1),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        net.model.fit(ds_train, callbacks=callbacks, epochs=500, steps_per_epoch=10)
        risks = net.model.predict(ds_train, steps=1)
        temp_all_risks = net.model.predict(ds_all)
        try:
            concordance = concordance_index(times[idx_train], np.exp(-risks[:, 0]), events[idx_train])
            runs += 1
        except:
            concordance = 0
        if concordance > temp_concordance:
            temp_concordance = concordance
            best_all_risks = temp_all_risks
            temp = np.exp(-temp_all_risks[:, 0]).argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp))
    test_ranks.append(ranks[idx_test])
    all_risks.append(best_all_risks)

results['2-neuron'] = [test_ranks, all_risks]

with open(cwd / 'figures' / 'msk_figure' / 'skcm.pkl', 'wb') as f:
    pickle.dump([test_idx, results], f)



