import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_survival':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_survival')]
import sys
sys.path.append(str(cwd))
import pickle
import numpy as np
from model.model import Encoders, NN
from model.layers import Losses
from model import utils
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-2], True)
tf.config.experimental.set_visible_devices(physical_devices[-2], 'GPU')

labels_to_use = ['BLCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'OV', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC']

data = pickle.load(open(cwd / 'files' / 'data.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
tmb_dict = {i[:12]: data[i][0] / (data[i][1] / 1e6) for i in data}

samples['tmb'] = samples.bcr_patient_barcode.apply(lambda x: tmb_dict.get(x, np.nan))
samples.dropna(axis=0, subset=['OS', 'OS.time', 'tmb'], inplace=True)

samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
samples['type'] = samples['type'].apply(lambda x: 'LUAD' if x == 'LUSC' else x)

results = {}
t = utils.LogTransform(bias=4, min_x=0)

for cancer in labels_to_use:
    print(cancer)
    df = samples.loc[samples['type'] == cancer]
    tmb = df['tmb'].values
    df = df.loc[tmb < np.percentile(tmb, 99)]
    tmb = t.trf(df.tmb.values)[:, np.newaxis]
    times = df['OS.time'].values[:, np.newaxis]
    events = df['OS'].values[:, np.newaxis]
    cancer_strat = np.zeros_like(df['tmb']) ##no cancer info
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
        
    results[cancer] = [test_idx, test_ranks, all_risks]

with open(cwd / 'figures' / 'fig3' / 'FCN_runs.pkl', 'wb') as f:
    pickle.dump(results, f)
