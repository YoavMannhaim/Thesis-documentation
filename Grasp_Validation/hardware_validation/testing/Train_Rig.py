"""
Trains grip-quality classifiers on the 10 recordings captured from the
current rig (2 live photoresistor channels), rather than on the original
4-sensor dataset, which no longer matches this hardware's range.

Split is by object, not by frame: train on the red sponge and the soap
holder, test on the glitter scrubber and the yellow sponge -- two objects
the model never sees. Frames from one recording are highly correlated, so a
random frame split would report memorisation as accuracy.

The two empty-gripper runs are split by time (empty_1 -> train, empty_2 ->
test), which also makes the test set carry the +35% baseline drift measured
between them -- deliberately, so the reported number reflects a model used
some time after calibration.
"""
import json, numpy as np, pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

W, STRIDE = 15, 3
CLASSES = ['no_grip', 'good_grip', 'bad_grip']

TRAIN = {'empty_1':'no_grip', 'redsponge_good':'good_grip', 'holder_good':'good_grip',
         'redsponge_bad':'bad_grip', 'holder_bad':'bad_grip'}
TEST  = {'empty_2':'no_grip', 'scrubber_good':'good_grip', 'yellow_good':'good_grip',
         'scrubber_bad':'bad_grip', 'yellow_bad':'bad_grip'}

def load(n):
    return pd.read_excel(f'captures/{n}.xlsx', header=None).iloc[:, :2].to_numpy(np.float32)

def frames(spec):
    X, y, src = [], [], []
    for name, lab in spec.items():
        a = load(name)
        X.append(a); y += [CLASSES.index(lab)]*len(a); src += [name]*len(a)
    return np.concatenate(X), np.array(y), np.array(src)

def windows(spec):
    X, y = [], []
    for name, lab in spec.items():
        a = load(name)
        for s in range(0, len(a)-W+1, STRIDE):
            X.append(a[s:s+W]); y.append(CLASSES.index(lab))
    return np.asarray(X), np.array(y)

def dense(nf):
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    i = Input((nf,)); h = Dropout(0.3)(Dense(64,activation='relu')(i))
    h = Dropout(0.3)(Dense(32,activation='relu')(h))
    m = Model(i, Dense(len(CLASSES),activation='softmax')(h))
    m.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def lstm(nf):
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    i = Input((W,nf)); x = Dropout(0.3)(LSTM(32)(i))
    x = Dropout(0.3)(Dense(16,activation='relu')(x))
    m = Model(i, Dense(len(CLASSES),activation='softmax')(x))
    m.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def report(tag, ytrue, ypred):
    acc = float(accuracy_score(ytrue, ypred))
    print(f"\n  {tag}: accuracy {acc:.4f}  (n={len(ytrue)})")
    print("    confusion (rows=true, cols=pred), order " + str(CLASSES))
    for row in confusion_matrix(ytrue, ypred, labels=range(len(CLASSES))):
        print("      ", row)
    return acc, classification_report(ytrue, ypred, target_names=CLASSES,
                                      output_dict=True, zero_division=0)

out = {'classes': CLASSES, 'train_files': TRAIN, 'test_files': TEST,
       'sensors': ['sensor_1','sensor_2'], 'window': W, 'stride': STRIDE}

print("="*60 + "\nSINGLE-FRAME (Dense)\n" + "="*60)
Xtr, ytr, _ = frames(TRAIN); Xte, yte, _ = frames(TEST)
sc = StandardScaler().fit(Xtr)
m = dense(2); m.fit(sc.transform(Xtr), ytr, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
p = np.argmax(m.predict(sc.transform(Xte), verbose=0), axis=1)
out['dense_accuracy'], out['dense_report'] = report("Dense, unseen objects", yte, p)
m.save('rig_dense_model.keras'); dump(sc, 'rig_dense_scaler.joblib')

print("\n" + "="*60 + "\nSEQUENCE (LSTM, 15-frame window)\n" + "="*60)
Xtr, ytr = windows(TRAIN); Xte, yte = windows(TEST)
n,w,f = Xtr.shape
sc2 = StandardScaler().fit(Xtr.reshape(-1,f))
Xtr_s = sc2.transform(Xtr.reshape(-1,f)).reshape(n,w,f)
Xte_s = sc2.transform(Xte.reshape(-1,f)).reshape(Xte.shape[0],w,f)
m2 = lstm(f); m2.fit(Xtr_s, ytr, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
p2 = np.argmax(m2.predict(Xte_s, verbose=0), axis=1)
out['lstm_accuracy'], out['lstm_report'] = report("LSTM, unseen objects", yte, p2)
m2.save('rig_lstm_model.keras'); dump(sc2, 'rig_lstm_scaler.joblib')

json.dump(out, open('rig_metrics.json','w'), indent=2)
print("\nmetrics -> rig_metrics.json")
